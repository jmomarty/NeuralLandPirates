from DCNN.DCNN import CNN as CNN
from DCNN.DCNN_with_W2V_transf import CNN as DeepCNN


def test_cnn_with_transf(vector_size=4):
    cnn = DeepCNN(vector_size=vector_size, num_labels=5)
    err = cnn.gradient_check()

    print 'found', err, 'errors...'
    assert err == 0


def test_cnn(vector_size=4):
    cnn = CNN(vector_size=vector_size, num_labels=5)
    err = cnn.gradient_check()

    print 'found', err, 'errors...'
    assert err == 0


if __name__ == "__main__":

    test_cnn()
    test_cnn_with_transf()
