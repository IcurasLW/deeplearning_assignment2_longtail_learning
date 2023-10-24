
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import random
from torchvision import transforms, datasets
import torch.utils.data
from losses import *
import os
from sklearn.model_selection import train_test_split



class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100




class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if len(x.size) == 2:
            x = transforms.Grayscale(num_output_channels=3)(x)

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)




def load_cifar10(args, imb_factor=0.01):
    mean_norm, std_norm = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norm,
                            std=std_norm),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norm,
                            std=std_norm),
    ])

    train_dataset = IMBALANCECIFAR10(root='./data', imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader



def load_cifar100(args, imb_factor=0.01):
    mean_norm, std_norm = [0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norm,
                            std=std_norm),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_norm,
                            std=std_norm),
    ])

    train_dataset = IMBALANCECIFAR100(root='./data', imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader


# def load_colitis(args):
#     train_image_dir = 'train_and_validation_sets'
#     test_image_dir = 'test_set'
#     mean_norm, std_norm = get_dataset_mean_and_std(train_image_dir)
#     transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                 transforms.RandomRotation((-180, 180)),
#                                 transforms.Resize((224, 224)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=mean_norm, std=std_norm)])
    
#     transform_test = transforms.Compose([transforms.Resize((224, 224)),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize(mean=mean_norm, std=std_norm)])
    
#     train_dataset = datasets.ImageFolder(root=train_image_dir, transform=transform_train)
#     test_dataset = datasets.ImageFolder(root=test_image_dir, transform=transform_test)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
#     return train_loader, test_loader



def load_caltech256(args):
    
    def modify_label(target, new_label_mapping):
        return new_label_mapping[target]

    dataset = datasets.Caltech256(root='data', download=True)
    label_counts = {}
    for _, label in dataset:
        # print(label)
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1


    new_key = [i for i in range(len(label_counts))]
    sorted_label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_label = list(sorted_label_counts.keys())
    mapping_label = {key:value for key, value in zip(sorted_label, new_key)}
    sorted_dataset = datasets.Caltech256(root='data', download=True, target_transform=lambda target: modify_label(target, mapping_label))


    train_transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomRotation((-180, 180)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    
    test_transforms = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # split the dataset
    dataset_indecies = [i for i in range(len(sorted_dataset))]
    label = sorted_dataset.y
    train_indices, test_indcies, _, _ = train_test_split(dataset_indecies, label, test_size=0.2, stratify=label)
    
    
    train_dataset = DatasetFromSubset(Subset(sorted_dataset, train_indices), transform=train_transforms)
    test_dataset = DatasetFromSubset(Subset(sorted_dataset, test_indcies), transform=test_transforms)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader




def get_dataset_mean_and_std(datapath):
    
    dataset = datasets.ImageFolder(datapath)
    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in dataset:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum(image[:, :, 0])
        G_total = G_total + np.sum(image[:, :, 1])
        B_total = B_total + np.sum(image[:, :, 2])

    R_mean = R_total / total_count
    G_mean = G_total / total_count
    B_mean = B_total / total_count


    R_total = 0
    G_total = 0
    B_total = 0
    total_count = 0
    
    for image, _ in dataset:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_total = G_total + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_total = B_total + np.sum((image[:, :, 2] - B_mean) ** 2)

    R_std = np.sqrt(R_total / total_count)
    G_std = np.sqrt(G_total / total_count)
    B_std = np.sqrt(B_total / total_count)

    return [R_mean / 255, G_mean / 255, B_mean / 255], [R_std / 255, G_std / 255, B_std / 255]



def prepare_data(args):
    if args.dataname == 'cifar10':
        train_loader, test_loader = load_cifar10(args)
    elif args.dataname == 'cifar100':
        train_loader, test_loader = load_cifar100(args)
    elif args.dataname == 'caltech256':
        train_loader, test_loader = load_caltech256(args)
    return train_loader, test_loader