import argparse

def opt_parse():
    parser = argparse.ArgumentParser(description='PyTorch Micro Expression Recognition')
    # basic setting
    parser.add_argument('--isTrain', dest='isTrain', action='store_true', default=True, help='If training the model')
    parser.add_argument('--label_map_trn', dest='label_map_trn', default='data/auxiliary/single_img_train.json', type=str, metavar='PATH',
                        help='File that contains label of each img for training')
    parser.add_argument('--label_map_tst', dest='label_map_tst', default='data/auxiliary/single_img_test.json', type=str, metavar='PATH',
                        help='File that contains label of each img for testing')
    parser.add_argument('--img_root', dest='img_root', default='../dataset/raw_imgs', type=str, metavar='PATH',
                        help='the img root dir')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15, 35],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-r', '--resize', dest='resize', type=int, default=128, help='resize img width to fixed shape')
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Gaussian sigma for mask')
    # Miscs
    parser.add_argument('-d', '--save_dir', default='checkpoint', type=str, metavar='PATH', help='path to save midterm parameters of model')
    parser.add_argument('--resume_epoch', default='None', type=str, metavar='PATH',
                        help='path to epoch to be resumed, if set to latest, resume latest model saved')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    return parser.parse_args()
