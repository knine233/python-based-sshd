import torch
import numpy as np
import os
import pickle
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import sys
from data.transform import train_transform, query_transform, Onehot, encode_onehot

ImageFile.LOAD_TRUNCATED_IMAGES = True




class Flickr25k(Dataset):

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = Flickr25k.TRAIN_DATA
            self.targets = Flickr25k.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Flickr25k.QUERY_DATA
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Flickr25k.RETRIEVAL_DATA
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError('Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    @staticmethod
    def init(root, num_query, num_train):
        # Read train.txt file
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            lines = f.readlines()
            Flickr25k.TRAIN_DATA = [line.strip().split()[0] for line in lines[:num_train]]
            Flickr25k.TRAIN_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines[:num_train]])

        # Read test.txt file
        with open(os.path.join(root, 'test.txt'), 'r') as f:
            lines = f.readlines()
            Flickr25k.RETRIEVAL_DATA = [line.strip().split()[0] for line in lines]
            Flickr25k.RETRIEVAL_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines])

        # Read database.txt file
        with open(os.path.join(root, 'database.txt'), 'r') as f:
            lines = f.readlines()
            Flickr25k.QUERY_DATA = [line.strip().split()[0] for line in lines[:num_query]]
            Flickr25k.QUERY_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines[:num_query]])

class COCO(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = COCO.TRAIN_DATA
            self.targets = COCO.TRAIN_TARGETS
        elif mode == 'query':
            self.data = COCO.QUERY_DATA
            self.targets = COCO.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = COCO.RETRIEVAL_DATA
            self.targets = COCO.RETRIEVAL_TARGETS
        else:
            raise ValueError('Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)
    
    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        # Read train.txt file
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            lines = f.readlines()
            COCO.TRAIN_DATA = np.asarray([line.strip().split()[0] for line in lines[:num_train]])
            COCO.TRAIN_TARGETS = np.asarray([list(map(int, line.strip().split()[1:])) for line in lines[:num_train]])

        # Read test.txt file
        with open(os.path.join(root, 'test.txt'), 'r') as f:
            lines = f.readlines()
            COCO.RETRIEVAL_DATA = np.asarray([line.strip().split()[0] for line in lines])
            COCO.RETRIEVAL_TARGETS = np.asarray([list(map(int, line.strip().split()[1:])) for line in lines])

        # Read database.txt file
        with open(os.path.join(root, 'database.txt'), 'r') as f:
            lines = f.readlines()
            COCO.QUERY_DATA = np.asarray([line.strip().split()[0] for line in lines[:num_query]])
            COCO.QUERY_TARGETS = np.asarray([list(map(int, line.strip().split()[1:])) for line in lines[:num_query]])

class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root, num_query, num_train):
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        base_folder = 'cifar-10-batches-py'

        data = []
        targets = []

        for file_name in data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = np.array(targets)

        # Sort by class
        sort_index = targets.argsort()
        data = data[sort_index, :]
        targets = targets[sort_index]

        # (num_query / number of class) query images per class
        # (num_train / number of class) train images per class
        query_per_class = num_query // 10
        train_per_class = num_train // 10

        # Permutate index (range 0 - 6000 per class)
        perm_index = np.random.permutation(data.shape[0] // 10)
        query_index = perm_index[:query_per_class]
        train_index = perm_index[query_per_class: query_per_class + train_per_class]

        query_index = np.tile(query_index, 10)
        train_index = np.tile(train_index, 10)
        inc_index = np.array([i * (data.shape[0] // 10) for i in range(10)])
        query_index = query_index + inc_index.repeat(query_per_class)
        train_index = train_index + inc_index.repeat(train_per_class)
        list_query_index = [i for i in query_index]
        retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index)), dtype=np.int64)

        # Split data, targets
        CIFAR10.QUERY_IMG = data[query_index, :]
        CIFAR10.QUERY_TARGET = targets[query_index]
        CIFAR10.TRAIN_IMG = data[train_index, :]
        CIFAR10.TRAIN_TARGET = targets[train_index]
        CIFAR10.RETRIEVAL_IMG = data[retrieval_index, :]
        CIFAR10.RETRIEVAL_TARGET = targets[retrieval_index]

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.data = CIFAR10.TRAIN_IMG
            self.targets = CIFAR10.TRAIN_TARGET
        elif mode == 'query':
            self.data = CIFAR10.QUERY_IMG
            self.targets = CIFAR10.QUERY_TARGET
        else:
            self.data = CIFAR10.RETRIEVAL_IMG
            self.targets = CIFAR10.RETRIEVAL_TARGET

        self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.FloatTensor(self.onehot_targets)

class NUS(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = NUS.TRAIN_DATA
            self.targets = NUS.TRAIN_TARGETS
        elif mode == 'query':
            self.data = NUS.QUERY_DATA
            self.targets = NUS.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = NUS.RETRIEVAL_DATA
            self.targets = NUS.RETRIEVAL_TARGETS
        else:
            raise ValueError('Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    @staticmethod
    def init(root, num_query, num_train):
        # Read train.txt file
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            lines = f.readlines()
            NUS.TRAIN_DATA = [line.strip().split()[0] for line in lines[:num_train]]
            NUS.TRAIN_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines[:num_train]])

        # Read test.txt file
        with open(os.path.join(root, 'test.txt'), 'r') as f:
            lines = f.readlines()
            NUS.RETRIEVAL_DATA = [line.strip().split()[0] for line in lines]
            NUS.RETRIEVAL_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines])

        # Read database.txt file
        with open(os.path.join(root, 'database.txt'), 'r') as f:
            lines = f.readlines()
            NUS.QUERY_DATA = [line.strip().split()[0] for line in lines[:num_query]]
            NUS.QUERY_TARGETS = torch.tensor([list(map(int, line.strip().split()[1:])) for line in lines[:num_query]])


def load_data(dataset_name, root, num_query, num_train, batch_size, num_workers):
    if dataset_name == 'flickr25k':
        DatasetClass = Flickr25k
        root = os.path.join(root, "flickr25k")
    elif dataset_name == 'coco':
        DatasetClass = COCO
        root = os.path.join(root, "coco")
    elif dataset_name == 'nus':
        DatasetClass = NUS
        root = os.path.join(root, "NUS-WIDE")
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset_name))

    DatasetClass.init(root, num_query, num_train)
    query_dataset = DatasetClass(root, 'query', query_transform())
    train_dataset = DatasetClass(root, 'train', train_transform())
    retrieval_dataset = DatasetClass(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader