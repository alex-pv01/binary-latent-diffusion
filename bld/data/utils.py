import collections

import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchvision import transforms
import numpy as np

from bld.data.helper_types import Annotation


def custom_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str" and elem_type.__name__ != "__string__":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int): 
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):
        return batch
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]
    
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def coco_collate(batch):
    """
    Collate function for coco dataset loader
    """
    new_batch = dict()
    new_captions = []
    new_images = []
    for elem in batch:
        img, captions = elem['image'], elem['captions']
        # Convert image to tensor
        img = transforms.ToTensor()(img)
        img = img.permute(1, 2, 0)
        # Duplicate image for each caption
        img = img.repeat(len(captions), 1, 1, 1)
        #print("img shape", img.shape)
        # Stack new captions and images into a list
        new_captions.extend(captions)
        new_images.append(img)
    # Convert list into tensor
    new_images = torch.cat(new_images)
    #print("new_images shape", new_images.shape)
    # Add to batch
    new_batch['image'] = new_images
    new_batch['captions'] = new_captions
            
    # Return collated batch
    return new_batch


def postprocess_image(image):
    image = (image + 1.0) * 127.5
    image = image.astype(np.uint8)
    print(image.shape)
    image = np.transpose(image, (1, 2, 0))
    image = transforms.ToPILImage()(image)
    print(image.size)
    return image

def save_image(image, name="test.png"):
    image = postprocess_image(image)
    # print image size
    print(image.size)
    image.save(name)