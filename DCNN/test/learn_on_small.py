import argparse
import DCNN.train_sentiment as train


if __name__ == '__main__':

    corpus = 'data/fake_corpus'
    vectors = 'data/fake_corpus'

    main_parser = argparse.ArgumentParser('Trains a sentimental CNN')
    main_parser.add_argument('--cat', type=int, default=5, help='choose number of classes')
    main_parser.add_argument('--deep', dest='model', action='store_const', const='deep', default='simple')
    main_parser.add_argument('--fully', dest='model', action='store_const', const='fully', default='simple')
    main_parser.add_argument('--kim', dest='model', action='store_const', const='kim', default='simple')
    main_parser.add_argument('--norm', dest='model', action='store_const', const='norm', default='simple')

    args, _ = main_parser.parse_known_args()

    fit_parser = argparse.ArgumentParser('Trains a sentimental CNN')
    fit_parser.add_argument('--out', default=None, dest='results_file')
    # fit_parser.add_argument('--save_at', default='test_results')
    fit_parser.add_argument('--method', default='adadelta')
    fit_parser.add_argument('--lr', default=0.1, type=float)
    fit_parser.add_argument('--information_freq', default=5, type=int)
    fit_parser.add_argument('--epoch', default=500, type=int)

    fit_args, _ = fit_parser.parse_known_args()

    print 'main args:', args
    print 'fit args:', fit_args

    train.main(corpus, vectors, args.cat, args.model, train_acc=True, **vars(fit_args))
