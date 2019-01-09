import torch
import os
import torch.backends.cudnn as cudnn

class BaseModel(object):
    ''' Base model with basic save and load method'''

    def __init__(self, opt):
        self.isTrain = opt.isTrain
        self.save_dir = opt.save_dir

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, epoch):
        path = os.path.join(self.save_dir, 'epoch_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict,
            'optimizer_state_dict': self.optimizer.state_dict
        }, path)

    def resume(self, epoch):
        path = os.path.join(self.save_dir, 'epoch_{}.pth'.format(epoch))
        checkpoint = torch.load(path)
        assert int(epoch) == int(checkpoint['epoch']), 'Error epochs don\'t match between checkpoint and its name, may be sth wrong in save method'
        print("--------------------------------------------------")
        print("-------------Epoch:{} state resumed-------------".format(checkpoint['epoch']))
        print("--------------------------------------------------\n")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.isTrain:
            self.train()
        else:
            self.eval()

    def init_cuda(self, *args):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True
        self.model = self.model.to(self.device)
        for arg in args:
            arg.cuda()
