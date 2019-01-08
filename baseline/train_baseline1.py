import torch
from torch.utils.data import DataLoader
from data import SingleImgDataset
from model import resnet50


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True
    dataloader =
