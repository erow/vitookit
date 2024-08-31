import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Callable

# from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image

import torch
import scipy.io as sio

from torchvision.datasets.folder import ImageFolder, VisionDataset, default_loader, IMG_EXTENSIONS, make_dataset, find_classes
from torchvision.datasets.utils import verify_str_arg

META_ROOT = "DevKit"
META_FILE = "meta.bin"
FOLDER = {"train":"train","val":"val"}



def serach_key(mydict,value):
    return (list(mydict.keys())[list(mydict.values()).index(value)])

class ImageNet(VisionDataset):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    .. note::
        Before using this class, it is required to download ImageNet 2012 dataset from
        `here <https://image-net.org/challenges/LSVRC/2012/2012-downloads.php>`_ and
        place the files ``ILSVRC2012_devkit_t12.tar.gz`` and ``ILSVRC2012_img_train.tar``
        or ``ILSVRC2012_img_val.tar`` based on ``split`` in the root directory.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None,
        ) -> None:
        root = self.root = os.path.expanduser(root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.split = verify_str_arg(split, "split", ("train", "val"))
        
        classid_to_wnid, wnid_to_classes = parse_meta_mat(os.path.join(self.root, META_ROOT))
        self.wnids = list(classid_to_wnid.values())
        self.wnids.sort()
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}
        
        self.classid_to_idx = { serach_key(classid_to_wnid,wnid): idx for idx, wnid in enumerate(self.wnids)}
        
        super().__init__(self.root, transform=transform, target_transform=target_transform)
        
        if self.split == "train":
            classes, class_to_idx = self.find_classes(self.split_folder)
            samples = self.make_dataset(self.split_folder, class_to_idx, extensions, is_valid_file)
        else:
            val_idcs = parse_val_groundtruth_txt(os.path.join(self.root, META_ROOT))
            imgs = os.listdir(self.split_folder)
            imgs.sort()
            samples = [(os.path.join(self.split_folder,file),self.classid_to_idx[val_idcs[img_id]]) for img_id, file in enumerate(imgs)]
            

        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

        
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        


    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, FOLDER[self.split])

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


    
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
    
def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, Tuple[str, ...]]]:
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(", ")) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes

def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
    file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
    with open(file) as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]

def parse_val_archive(
    root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
) -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """

    val_root = os.path.join(root, folder)

    images = sorted(os.path.join(val_root, image) for image in os.listdir(val_root))

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))

IN100SUBSET = ['n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406', 'n02091831', 'n04517823', 'n04589890', 'n03062245', 'n01773797', 'n01735189', 'n07831146', 'n07753275', 'n03085013', 'n04485082', 'n02105505', 'n01983481', 'n02788148', 'n03530642', 'n04435653', 'n02086910', 'n02859443', 'n13040303', 'n03594734', 'n02085620', 'n02099849', 'n01558993', 'n04493381', 'n02109047', 'n04111531', 'n02877765', 'n04429376', 'n02009229', 'n01978455', 'n02106550', 'n01820546', 'n01692333', 'n07714571', 'n02974003', 'n02114855', 'n03785016', 'n03764736', 'n03775546', 'n02087046', 'n07836838', 'n04099969', 'n04592741', 'n03891251', 'n02701002', 'n03379051', 'n02259212', 'n07715103', 'n03947888', 'n04026417', 'n02326432', 'n03637318', 'n01980166', 'n02113799', 'n02086240', 'n03903868', 'n02483362', 'n04127249', 'n02089973', 'n03017168', 'n02093428', 'n02804414', 'n02396427', 'n04418357', 'n02172182', 'n01729322', 'n02113978', 'n03787032', 'n02089867', 'n02119022', 'n03777754', 'n04238763', 'n02231487', 'n03032252', 'n02138441', 'n02104029', 'n03837869', 'n03494278', 'n04136333', 'n03794056', 'n03492542', 'n02018207', 'n04067472', 'n03930630', 'n03584829', 'n02123045', 'n04229816', 'n02100583', 'n03642806', 'n04336792', 'n03259280', 'n02116738', 'n02108089', 'n03424325', 'n01855672', 'n02090622']

class ImageNet100(ImageNet):
    def __init__(self, root: str, split: str = "train", transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, loader: Callable[[str], Any] = default_loader, is_valid_file: Callable[[str], bool] | None = None) -> None:
        super().__init__(root, split, transform, target_transform, loader, is_valid_file)
        
        self.samples = [s for s in self.samples if self.wnids[s[1]] in IN100SUBSET]

class ImageNetV2(ImageFolder):
    def __init__(self,
        root: str, 
        split:str = "matched-frequency",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ):
        root = self.valid_splits(root, split)
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader = loader,
        )
    
    def valid_splits(self,root,split):
        file_link = {
            "matched-frequency":"imagenetv2-matched-frequency.tar.gz",
            "threshold0.7":"imagenetv2-threshold0.7.tar.gz",
            "topimages":"imagenetv2-top-images.tar.gz"
        }
        folder = file_link[split].replace(".tar.gz","-format-val")
        target = os.path.join(root,folder)
        if os.path.exists(target):
            return target
        print(f"Downloading {split} from huggingface_hub")
        from huggingface_hub import hf_hub_download
        cache_file = hf_hub_download("vaishaal/ImageNetV2", file_link[split],repo_type="dataset")
        print(f"Extracting {cache_file} to {root}")
        import tarfile
        with tarfile.open(cache_file) as tar:
            tar.extractall(path=root)
        return target

        