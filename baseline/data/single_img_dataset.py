import os, json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from PIL import Image

class SingleImgDataset(Dataset):
    """ For single img (Not img Sequence) """

    def __init__(self, _file, root_dir, opt=None):
        self.transform = self.get_transforms(opt)
        self.data = json.load(open(_file))
        self.root_dir = root_dir

    def __getitem__(self, idx):
        record = self.data[idx]
        path = os.path.join(self.root_dir, record['path'])
        img = Image.open(path)
        print(type(self.transform))
        img = self.transform(img)
        label = record['label']
        return {'img': img, "label": label, 'path': path}


    def __len__(self):
        return len(self.data)

    def get_transforms(self, opt):
        transform_list = []
        # , transforms.CenterCrop(opt['resize'])
        transform_list += [transforms.Resize(opt['resize'], interpolation=2), transforms.RandomHorizontalFlip()]
        transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def name(self):
        return 'Single Img Dataset'
