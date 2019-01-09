from .resnet import resnet50
from .base_model import BaseModel
import torch.optim as optim
import torch.nn.functional as NF

class BaselineModel(BaseModel):
    ''' for simple resnet model '''

    def __init__(self, opt):
        super().__init__(opt)
        self.model = resnet50()
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999))
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, opt.schedule, opt.gamma)
        if opt.resume_epoch != 'None':
            self.resume(opt.resume_epoch)
        self.init_cuda()
        self.criterion = NF.cross_entropy

    def set_input(self, data):
        self.x = data['img'].cuda()
        self.label = data['label'].cuda()
        self.path = data['path']

    def forward(self):
        self.pred = self.model(self.x)

    def backward(self):
        self.loss = self.criterion(self.pred, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def run_one_batch(self, data):
        self.scheduler.step()
        self.set_input(data)
        self.forward()
        if self.isTrain:
            self.backward()
        return self.pred

    def get_current_loss(self):
        return self.loss
