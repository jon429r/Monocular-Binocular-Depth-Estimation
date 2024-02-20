## This file is a modified version of the original torchvision.datasets.kitti.py file

import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

import torch

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive



class Kitti(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: int 
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """



        image = Image.open(self.images[index])
        target = self._parse_target(index) if self.train else [{
                        # map this type value to the label_mapping

                        "type": int(0),
                        "truncated": float(0),
                        "occluded": int(0),
                        "alpha": float(0),
                        "bbox": [float(0)],
                        "dimensions": [float(0)],
                        "location": [float(0)],
                        "rotation_y": float(0),
                    }]
        

        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target
    
    def target_to_tensor(self, target):
        target["type"] = torch.tensor(target["type"])
        target["truncated"] = torch.tensor(target["truncated"])
        target["occluded"] = torch.tensor(target["occluded"])
        target["alpha"] = torch.tensor(target["alpha"])
        target["bbox"] = torch.tensor(target["bbox"])
        target["dimensions"] = torch.tensor(target["dimensions"])
        target["location"] = torch.tensor(target["location"])
        target["rotation_y"] = torch.tensor(target["rotation_y"])

        return target

    def _parse_target(self, index: int) -> List:
        target = []

        label_mapping = {
            'Pedistrian': 1,
            'Car': 2,
            'Cyclist': 3,
            'Truck': 4,
            'DontCare': 5,
            'Misc': 6,
            'Other': 0,
        }
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": label_mapping.get(line[0], 0),
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )

            inp.seek(0)
        
            # Count the number of lines in the file
            length = sum(1 for line in content)
            while length < 32:
                target.append(
                    {
                        "type": 0,
                        "truncated": float(0),
                        "occluded": int(0),
                        "alpha": float(0),
                        "bbox": [float(0)],
                        "dimensions": [float(0)],
                        "location": [float(0)],
                        "rotation_y": float(0),
                    }
                )
                length += 1

        return target

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def download(self) -> None:
        """Download the KITTI data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self._raw_folder, exist_ok=True)

        # download files
        for fname in self.resources:
            download_and_extract_archive(
                url=f"{self.data_url}{fname}",
                download_root=self._raw_folder,
                filename=fname,
            )