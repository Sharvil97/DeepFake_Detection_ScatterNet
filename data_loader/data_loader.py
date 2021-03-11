import os
import sys
import cv2
import torch
import os.path
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.vision import VisionDataset
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded,
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray)


sys.path.append(os.path.realpath('..'))

# class GeneralTrainDataset(Dataset):
#     """
#     Give the required dataset to load.
#     The following formart is required:
#     data/<name-of-the-dataset>/ -> (./train -> ./real , ./fake), (./val -> ./real , ./fake), (./test -> ./real , ./fake)
#     the test folder contains videos as we do evaluations on the windows as opposed to frames.
#     """
#     def __init__(self, data, img_size, normalization, augmentations):
#         self.img_size = img_size
#         self.normalization = normalization
#         self.augmentations = augmentations

        # if data == "FaceForensics":
        #     self.data = r"../data/FaceForensics/train"
        # if data == "FaceForensics++":
        #     self.data = r"../data/FaceForensics++/train"
        # if data == "CelebDF":
        #     self.data = r"../data/CelebDF/train"
        # if data == "GoogleDFD":
        #     self.data = r"../data/GoogleDFD/train"
        # if data == "FaceHQ":
        #     self.data = r"../data/FaceHQ/train"  
        # if data == "DFDC":
        #     self.data =   r"../data/DFDC/train"   
        # if data == "DeeperForensics":
        #     self.data =   r"../data/DeeperForensics/train"   
        # if data == "UADFV":
        #     self.data =   r"../data/UADFV/train"     
        
#         all_imgs = os.listdir(self.data)
#         self.total_imgs = natsort.natsorted(all_imgs)
    
#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.data_dir, self.total_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image

#     def __len__(self):
#         return len(self.total_imgs)



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None,augmentations=True):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if augmentations:     
            """
            #TODO: Change implementation for ImageCompression
            """                              
            self.augmentations = Compose({
                Resize(256, 256),
                OneOf([RandomBrightnessContrast(), HueSaturationValue(),  GaussianBlur(), FancyPCA(),
                GaussNoise(), ToGray()]),
                HorizontalFlip(),
                ShiftScaleRotate(),
            })
        else:
            self.augmentations = Compose({
                Resize(256, 256),})




        self.img_size = 256 #img_size
        self.normalization = "imagenet"

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.augmentations:
                sample = self.augmentations(image=sample)['image']
        else:
            # no augmentation during validation or test, just resize to fit DNN input
#             augmentations = Resize(
#                 width=self.img_size, height=self.img_size)
            sample = self.augmentations(image=sample)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            # img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            sample = sample.float() / 255.0
            # normalize
        sample = transforms.ToTensor()(sample)
        if self.normalization == "xception":
            # normalize by xception stats
            transform = transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif self.normalization == "imagenet":
            # normalize by imagenet stats
            transform = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        sample = transform(sample)
        

        return sample, target


    def __len__(self):
        return len(self.samples)



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrain(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, data, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, augmentations=True):
        if data == "FaceForensics":
            root = r"data/FaceForensics/train"
        if data == "FaceForensics++":
            root = r"data/FaceForensics++/train"
        if data == "CelebDF":
            root = r"data/CelebDF/train"
        if data == "GoogleDFD":
            root = r"data/DeepFakeDetection/train"
        if data == "FaceHQ":
            root = r"data/FaceHQ/train"  
        if data == "DFDC":
            root =   r"data/DFDC/train"   
        if data == "DeeperForensics":
            root =   r"data/DeeperForensics/train"   
        if data == "UADFV":
            root =   r"data/UADFV/train"  
        if data == "NeuralTexture":
            root = r"data/NeuralTexture/train"
        if data == "Deepfakes":
            root = r"data/Deepfakes/train"
        if data == "FaceSwap":
            root = r"data/FaceSwap/train"
        if data == "FaceShifter":
            root = r"data/FaceShifter/train"
        if data == "Face2Face":
            root = r"data/Face2Face/train"
        super(ImageFolderTrain, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
#                                           img_size=img_size,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, augmentations=augmentations)
        self.imgs = self.samples


class ImageFolderVal(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, data,  transform=None, target_transform=None,
                 loader=default_loader,  is_valid_file=None, augmentations=False):
        if data == "FaceForensics":
            root = r"data/FaceForensics/val"
        if data == "FaceForensics++":
            root = r"/data/FaceForensics++/val"
        if data == "CelebDF":
            root = r"data/CelebDF/val"
        if data == "GoogleDFD":
            root = r"data/DeepFakeDetection/val"
        if data == "FaceHQ":
            root = r"data/FaceHQ/val"  
        if data == "DFDC":
            root =   r"data/DFDC/val"   
        if data == "DeeperForensics":
            root =   r"data/DeeperForensics/val"   
        if data == "UADFV":
            root =   r"data/UADFV/val"     
        if data == "NeuralTexture":
            root = r"data/NeuralTexture/val"
        if data == "Deepfakes":
            root = r"data/Deepfakes/val"
        if data == "FaceSwap":
            root = r"data/FaceSwap/val"
        if data == "FaceShifter":
            root = r"data/FaceShifter/val"
        if data == "Face2Face":
            root = r"data/Face2Face/val"
        super(ImageFolderVal, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
#                                           img_size=img_size,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, augmentations=augmentations)
        self.imgs = self.samples
        
