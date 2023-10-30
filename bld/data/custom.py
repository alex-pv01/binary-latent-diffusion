from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoCaptions
import torchvision.transforms as transform
import albumentations
import numpy as np

from bld.data.base import ImagePaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class COCODataset(CustomBase):
    def __init__(self, size, training_images_folder_path, ann_json, random_crop=False, ):
        super().__init__()
        self.data = CocoCaptions(root=training_images_folder_path, annFile=ann_json, transform=transform.PILToTensor())
        self.size = size
        self.random_crop = random_crop
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def preprocess_image(self, image):
        # Convert tensor to PIL image
        image = transform.ToPILImage()(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
    
    def __getitem__(self, index):
        item = dict()
        item["image"] = self.preprocess_image(self.data[index][0])
        item["captions"] = self.data[index][1]
        return item