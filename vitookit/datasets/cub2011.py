# copy from https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    seg_url = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    seg_filename = 'segmentations.tgz'
    seg_md5 = '56989585210501b1f12e9e5d8ad97edd'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        classes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                             sep=' ', names=['class_id', 'class_name'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.classes = classes
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            
    def _load_attributes(self):
        class_attributes_file = os.path.join(self.root, 'CUB_200_2011', 'attributes', 'class_attribute_labels_continuous.txt')
        class_attributes = pd.read_csv(class_attributes_file, sep=' ', header=None)        
        self.class_attributes = class_attributes
        

    def _load_segmentations(self):
        segmentations = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'segmentations', 'segmentations.txt'),
                                    sep=' ', names=['img_id', 'segmentation'])
        self.segmentations = segmentations

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print("Downloading CUB 2011 dataset")
            print("Please download the dataset from the following URL manually and place it in the root directory")
            print('https://data.caltech.edu/records/65de6-vp158')

        download_url(self.url, self.root, self.filename, self.tgz_md5)
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)
        # download_url(self.seg_url, self.root, self.seg_filename, self.seg_md5)
        # with tarfile.open(os.path.join(self.root, self.seg_filename), "r:gz") as tar:
        #     tar.extractall(path=self.root)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target