import numpy as np # Keep numpy and torch imports at top level
import torch

# Attempt to import monotonic_align and provide fallbacks
try:
    from monotonic_align import maximum_path as original_maximum_path_imported
    from monotonic_align import mask_from_lens as original_mask_from_lens_imported
    from monotonic_align.core import maximum_path_c as original_maximum_path_c_imported
    MONOTONIC_ALIGN_AVAILABLE = True
    # Assign them to the names used by the original code if they were directly used
    # For example, if a function in this file was also named maximum_path, it would need aliasing.
    # However, the primary `maximum_path` in this file is defined *later*.
    # We will handle `maximum_path_c` inside the `maximum_path` function.
    mask_from_lens = original_mask_from_lens_imported # if used directly
except ImportError:
    print("Warning: monotonic_align package not found or failed to import. Related functionality will be dummied.")
    MONOTONIC_ALIGN_AVAILABLE = False
    original_maximum_path_imported = None
    original_mask_from_lens_imported = None
    original_maximum_path_c_imported = None
    mask_from_lens = None # Dummy assignment

    def dummy_mask_from_lens(*args, **kwargs):
        print("Warning: Called dummy_mask_from_lens due to monotonic_align import failure.")
        raise NotImplementedError("mask_from_lens is unavailable because monotonic_align failed to import.")
    
    if mask_from_lens is None: # If the import failed, assign the dummy
        mask_from_lens = dummy_mask_from_lens

# Ensure other imports that were below monotonic_align are still present
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  if not MONOTONIC_ALIGN_AVAILABLE:
    print("Warning: Called dummy_maximum_path logic due to monotonic_align import failure.")
    # Return a zero tensor of the expected shape path: [b, t_t, t_s]
    return torch.zeros_like(neg_cent, dtype=torch.int32, device=neg_cent.device)

  # Proceed with original logic if monotonic_align is available
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent_np = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path_np = np.ascontiguousarray(np.zeros(neg_cent_np.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  
  # Use the imported maximum_path_c
  original_maximum_path_c_imported(path_np, neg_cent_np, t_t_max, t_s_max)
  return torch.from_numpy(path_np).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    