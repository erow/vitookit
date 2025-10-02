import os, json
import gin

from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import ImageFolder, default_loader

def img_loader(path):
    try:
        img = default_loader(path)
    except:
        img = None
    return img

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index
    

@gin.configurable()
def build_dataset(args, is_train, trnsfrm=None,):

    if trnsfrm is None:
        trnsfrm = build_transform(is_train, args)
    
    tfm = trnsfrm
    
    if 'data_path' in args.__dict__:
        args.data_location = args.data_location
    
    data_set = args.data_set#.upper()
    if data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = datasets.OxfordIIITPet(args.data_location, split=split, transform=tfm,download=True)
        nb_classes = 37
    elif data_set == 'Folder':
        dataset = datasets.ImageFolder(args.data_location, transform=tfm,loader=img_loader)
        nb_classes = len(dataset.classes)
    elif data_set == 'DA': # domestic animals
        split = 'train' if is_train else 'validation'
        dataset = datasets.ImageFolder(os.path.join(args.data_location,split), transform=tfm,loader=img_loader)
        wnids = ['n02091635', 'n02098286', 'n02112350', 'n02110806', 'n02095889'] + ['n02123159', 'n02123394', 'n02124075', 'n02123045', 'n02123597']
        indices = [dataset.classes.index(label) for label in wnids]
        dataset.samples = [(path,indices.index(label)) for path,label in dataset.samples if label in indices]
        dataset.classes = wnids
        nb_classes = 10
    elif data_set in ['IN1K','IN100']:
        split = 'train' if is_train else 'validation'
        dataset = datasets.ImageFolder(os.path.join(args.data_location,split), transform=tfm)
        nb_classes = len(dataset.classes)
    elif data_set == 'IN1Kv2':
        dataset = datasets.ImageFolder(args.data_location, transform=tfm)
        true_classes = dataset.classes
        dataset.samples = [(path,int(true_classes[label])) for path,label in dataset.samples ]
        nb_classes = len(dataset.classes)
    elif data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_location,is_train,transform=tfm,download=True)
        nb_classes = 10
        
    elif data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_location,is_train,transform=tfm,download=True)
        nb_classes = 100
        
    elif data_set == 'Cars':
        dataset = datasets.StanfordCars(args.data_location,'train' if is_train else 'test',transform=tfm,download=True)
        nb_classes = 196
    
    elif data_set == 'Flowers':
        dataset = datasets.Flowers102(args.data_location,'train' if is_train else 'test',transform=tfm,download=True)
        nb_classes = 102
    
    elif data_set == 'Aircraft':
        dataset = datasets.FGVCAircraft(args.data_location,'trainval' if is_train else 'test',transform=tfm,download=True)
        nb_classes = len(dataset.classes)

    elif data_set == 'STL':
        split = 'train' if is_train else 'test'
        dataset = datasets.STL10(args.data_location,split,transform=tfm,download=True)
        nb_classes = 10

    elif data_set == 'ominiglot':
        trnsfrm.transforms.insert(-2,transforms.Grayscale(num_output_channels=3))
        dataset = datasets.Omniglot(args.data_location,transform=tfm,download=True)
        nb_classes = 1623

    elif data_set == 'INAT':
        dataset = INatDataset(args.data_location, train=is_train, year=2018,
                              transform=tfm)
        nb_classes = dataset.nb_classes

    elif data_set == 'Food':
        split = 'train' if is_train else 'test'
        dataset=datasets.Food101(args.data_location, split,tfm,download=True)
        nb_classes=101
        
    elif data_set == 'CoarseIN493':
        coarse_map = json.load(open(os.path.join(args.data_location,"coarse_map.json")))
        split = 'train' if is_train else 'validation'
        dataset = datasets.ImageFolder(os.path.join(args.data_location,split), transform=tfm)        
        coarse_labels = [coarse_map[k] for k in dataset.class_to_idx]
        def target_transform(target):
            return coarse_labels[target]
        dataset.target_transform = target_transform
        nb_classes = 31

    elif data_set =='DTD':
        split = 'train' if is_train else 'test'
        dataset = datasets.DTD(args.data_location,split, 
                            transform=tfm, download=True)
        nb_classes = 47
    
    elif data_set == 'SUN397':
        split = 'train' if is_train else 'test'
        dataset = datasets.SUN397(args.data_location, 
                            transform=tfm, download=True)
        nb_classes = 397

    elif data_set == 'CUB200':
        from vitookit.datasets.cub2011 import Cub2011
        dataset = Cub2011(args.data_location, is_train,
                            transform=tfm, download=True)
        nb_classes = 200
        
    elif data_set == 'MNIST':
        dataset = datasets.MNIST(args.data_location, is_train, transform=tfm, download=True)
        nb_classes = 10
        
    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(data_set))
        raise NotImplementedError(data_set,args.data_location)
        
    return dataset, nb_classes

def get_supported_datasets():
    datasets = [
        "DTD",
        "SUN397",
        "CUB200",
        "Food",
        "Flowers",
        "Cars",
        "Pets",
        "Aircraft",
        "INAT",
        "STL",
        "CIFAR10",
        "CIFAR100",
        "ominiglot",
        "ImageNet",
        "Folder",
    ]
    return datasets

@gin.configurable(denylist=['is_train','args'])
def build_transform(is_train, args,
                    mean = IMAGENET_DEFAULT_MEAN,
                    std = IMAGENET_DEFAULT_STD,
                    **kwargs):
    
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = timm.data.create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
            **kwargs
        )
        return transform
    else:
        # eval transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)



class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.root = root
        self.target_transform = target_transform
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))
