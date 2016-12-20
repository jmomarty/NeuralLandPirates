function Train(data_f)

addpath(genpath('./autoDif'))

load(data_f,'train','train_lbl', 'valid', 'valid_lbl',  'test', 'test_lbl', 'size_vocab','sent_length', 'vocab_emb', 'index');


%% ASSIGN P AND indices
p(1) = 5;          disp(strcat('Size word vectors:',num2str(p(1))));
p(2) = sent_length; disp(strcat('Max sent length:',num2str(p(2))));
p(3) = 3;           disp(strcat('Number feat maps in first layer:',num2str(p(3))));
p(5) = 4;           disp(strcat('Number feat maps in second layer:', num2str(p(5))));
p(37) = 2;          disp(strcat('Number feat maps in third layer:', num2str(p(37))));
p(4) = 6;           disp(strcat('Size of kernel in first layer:', num2str(p(4))));
p(6) = 5;           disp(strcat('Size of kernel in second layer:', num2str(p(6))));
p(36) = 3;          disp(strcat('Size of kernel in third layer:', num2str(p(36))));
p(8) = 0;           disp(strcat('Using relu:',num2str(p(8))));
p(9) = 6;           disp(strcat('Number of output classes:',num2str(p(9))));
p(10) = 3;          disp(strcat('Number of conv layers being used (1 or 2 or 3):',num2str(p(10))));
p(7) = 2;           disp(strcat('TOP POOLING width:',num2str(p(7)))); 
p(12) = 0;          disp(strcat('Folding in first layer:', num2str(p(12))));
p(13) = 0;          disp(strcat('Folding in second layer:', num2str(p(13))));
p(35) = 0;          disp(strcat('Folding in third layer:',num2str(p(35))));
p(30) = size_vocab; disp(strcat('Size vocab (and pad):',num2str(p(30))));
p(32) = 1;          disp(strcat('Word embedding learning ON:',num2str(p(32))));
p(33) = 199;        disp(strcat('if emb learn ON, after how many epochs OFF:',num2str(p(33))));
p(34) = 1;          disp(strcat('use preinitialized vocabulary:',num2str(p(34))));

p(40) = 1;          disp(strcat('Dropout at Projection Sentence matrix:',num2str(p(40))));
p(41) = 1;          disp(strcat('Dropout at First layer:',num2str(p(40))));
p(42) = 1;          disp(strcat('Dropout at Second layer:',num2str(p(40))));
p(43) = 1;          disp(strcat('Dropout at Third layer:',num2str(p(40))));

p(51) = 3;          disp(strcat('Local Connections Span at First layer:',num2str(p(40))));
p(52) = 3;          disp(strcat('Local Connections Span at Second layer:',num2str(p(40))));
p(53) = 3;          disp(strcat('Local Connections Span at Third layer:',num2str(p(40))));

%
%
disp(' ');
p(20) = 1e-4;       disp(strcat('Reg E (word vectors):',num2str(p(20))));
p(21) = 3e-5;       disp(strcat('Reg 1 (first conv layer):',num2str(p(21))));
p(22) = 3e-6;       disp(strcat('Reg 2 (second conv layer):',num2str(p(22))));
p(23) = 1e-5;       disp(strcat('Reg 3 (third conv layer):',num2str(p(23))));
p(24) = 1e-4;       disp(strcat('Reg Z (classification layer):',num2str(p(24))));
%
%
%
p(31) = 0;         disp(strcat('GPU and SINGLE on:',num2str(p(31))));
%% MASKS LENGTHS
%p(25) max_sent_lenth+kernel_1-1 
%p(26) max_layer_1_lenth+kernel_2-1 
%p(27) max_layer_2_lenth+kernel_3-1


[train_msk, valid_msk, test_msk, p] = Masks(train, train_lbl, valid, valid_lbl, test, test_lbl, p);
CR = RCTM(p);
if p(34) %if use external vocabulary
    CR.E = vocab_emb(1:p(1),:);
    CR.E(:,p(30)) = zeros(size(CR.E,1),1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPRESENTATION SIZES
disp(' ')
disp(['Layer 1: ' num2str(p(3)) ' maps of depth ' num2str(p(1)/(p(12)*1+1))])
if p(10) >= 2, disp(['Layer 2: ' num2str(p(5)) ' maps of depth ' num2str(p(1)/((p(12)+1)*(p(13)+1)))]), end
if p(10) == 3, disp(['Layer 3: ' num2str(p(37)) ' maps of depth ' num2str(p(1)/((p(12)+1)*(p(13)+1)*(p(35)+1)))]), end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%


train_lbl = train_lbl(:,1); %getting rid of length information for sentences
test_lbl = test_lbl(:,1);
valid_lbl = valid_lbl(:,1);


if p(31) %if use GPU
    train = single(train);
    train_lbl = single(train_lbl);
    valid = single(valid);
    valid_lbl = single(valid_lbl);
    test = single(test);
    test_lbl = single(test_lbl);
    if p(34), vocab_emb = single(vocab_emb); end
    train_msk = logical(train_msk);
    test_msk = logical(test_msk);
    valid_msk = logical(valid_msk); 
    p = single(p);
end


%% TRAINING

[X, decodeInfo] = param2stack(CR.E, CR.one, CR.one_b, CR.two, CR.two_b, CR.three, CR.three_b, CR.Z, [], [], p);
disp(strcat('Total number of parameters: ', num2str(numel(X))));
disp(' ');



gamma = 0.1;
batchsize = 3;
maxEpochs = 199;
disp(strcat('Learning rate:',num2str(gamma)));
disp(strcat('Batch size:',num2str(batchsize)));
disp(strcat('Max Number of epochs:',num2str(maxEpochs)));

num_batch_epochs = floor(size(train,1)/(batchsize)); %leaves last batch out at an iteration
indices = kron(1:p(1),ones(1,batchsize*p(2)+1)).'; %adding one value for consistent size of E_df

valid_batch = reshape(valid',1,[]);
test_batch = reshape(test',1,[]);

if p(31) %if GPU
    indices = single(indices);
    
    valid_batch = gpuArray(valid_batch);
    valid_lbl = gpuArray(valid_lbl);
    valid_msk = gpuArray(valid_msk);
	
    test_batch = gpuArray(test_batch);
    test_lbl = gpuArray(test_lbl);
    test_msk = gpuArray(test_msk);

    X = gpuArray(single(X));
    decodeInfo = gpuArray(single(decodeInfo));
end


H = 0;
AC = 10;

for i=1:maxEpochs
     permut = randperm(size(train,1));
     train= train(permut,:);
     train_lbl = train_lbl(permut);
     train_msk = train_msk(permut,:);
     gradient_hist = zeros(length(X),1); %TODO NOTE: resetting could be done less often. Parametrize this
    
    disp(strcat('Epoch:', num2str(i)));
    
    if i>p(33)
       p(32) = 0; %turn off embedding learning after a few epochs if p(32)==1
    end
    
    if p(31) %if GPU
        p = gpuArray(p);
        gamma = gpuArray(gamma);
        batchsize = gpuArray(batchsize);
        train = gpuArray(train);
        train_lbl = gpuArray(train_lbl);
        train_msk = gpuArray(logical(train_msk));
        gradient_hist = gpuArray(single(gradient_hist));
    end
   
    for j=1:num_batch_epochs
        minibatch = reshape(train((j-1)*batchsize+1:j*batchsize,:)',1,[]);%fixed size batches
        labels = train_lbl((j-1)*batchsize+1:j*batchsize);
        mini_msk = train_msk((j-1)*batchsize+1:j*batchsize,:);
        
        if 0, fastDerivativeCheck(@CostFunction,X,1,2, decodeInfo, minibatch, labels, mini_msk, indices, p); end
        [cost,grad]=CostFunction(X,  decodeInfo, minibatch, labels, mini_msk, indices, p);
        
        if j <= 100 %Only print PPL at the beginning
            disp(['J:' num2str(j) ' PPL: ' num2str(exp(cost))]);
        end
        
        gradient_hist = gradient_hist + grad.^2;
        sq = sqrt(gradient_hist);
        sq(sq~=0) = gamma./sq(sq~=0);
        X = X-sq.*grad;
        
        %Accuracies
        if mod(j,AC) == 0 || j==num_batch_epochs
            disp(['I:' num2str(i) ' J:' num2str(j) ' PPL: ' num2str(exp(cost))]);
            batch_size = 200;
            v_acc = 0;
            for b=1:ceil(length(valid_lbl)/batch_size)
                v_acc = v_acc + Accuracy(X, decodeInfo, valid_batch(((b-1)*batch_size*p(2))+1:min(b*batch_size*p(2),end)), valid_lbl((b-1)*batch_size+1:min(b*batch_size,end)), valid_msk((b-1)*batch_size+1:min(b*batch_size,end),:), p);
            end  
            v_acc = v_acc/length(valid_lbl);
            disp(['Validation accuracy:', num2str(v_acc)]);
            disp(['Average Parameter Weight: ', num2str(sum(abs(X))/length(X))]);
            if v_acc > H
                H = v_acc;
                save('QA3','X','decodeInfo','v_acc', 'i', 'j','p');
            end
        end 
    end
end








