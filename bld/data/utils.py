import collections
import os 
import tarfile
import urllib
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
import tqdm

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
    elif isinstance(elem, string_classes):
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