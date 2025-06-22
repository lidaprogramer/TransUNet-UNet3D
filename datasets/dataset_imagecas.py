"""ImageCas 2‑D slice dataset – compatible with trainer_ImageCas.

* Expects list files produced by **preprocess_imagecas.py**
  ├── train.txt          (one slice name per line, *without* .npz)
  └── test_vol.txt       (validation slices)

* Expects slices stored as compressed .npz under
      <base_dir>/train_processed_224   (for split == 'train')
      <base_dir>/val_processed_224     (for split != 'train')

Each .npz has keys  `image` (H×W float) and `label` (H×W int).
The class applies the same random augmentations as dataset_penguin.
"""

from __future__ import annotations

import os, random, numpy as np, torch, h5py
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
#  Simple augmentations
# -----------------------------------------------------------------------------

def _random_rot_flip(img: np.ndarray, lbl: np.ndarray):
    k = np.random.randint(0, 4)
    img  = np.rot90(img,  k)
    lbl  = np.rot90(lbl,  k)
    axis = np.random.randint(0, 2)
    img  = np.flip(img,  axis=axis).copy()
    lbl  = np.flip(lbl,  axis=axis).copy()
    return img, lbl


def _random_rotate(img: np.ndarray, lbl: np.ndarray):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img,  angle, order=0, reshape=False)
    lbl = ndimage.rotate(lbl, angle, order=0, reshape=False)
    return img, lbl


class RandomGenerator:
    """Match Penguin’s on‑the‑fly augmentation/resize pipeline."""
    def __init__(self, output_size: tuple[int, int]):
        self.output_size = output_size

    def __call__(self, sample):
        img, lbl = sample['image'], sample['label']

        if random.random() > 0.5:
            img, lbl = _random_rot_flip(img, lbl)
        elif random.random() > 0.5:
            img, lbl = _random_rotate(img, lbl)

        H, W = img.shape
        if (H, W) != self.output_size:
            zoom_f = (self.output_size[0] / H, self.output_size[1] / W)
            img = zoom(img,  zoom_f, order=3)
            lbl = zoom(lbl,  zoom_f, order=0)

        img_t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # C,H,W
        lbl_t = torch.from_numpy(lbl.astype(np.float32))
        return {'image': img_t, 'label': lbl_t.long(), 'case_name': sample['case_name']}


# -----------------------------------------------------------------------------
#  Dataset class
# -----------------------------------------------------------------------------

class ImageCas_dataset(Dataset):
    """ImageCas .npz slice loader (train/val)."""
    def __init__(self, base_dir: str, list_dir: str, split: str = 'train', transform=None, positive_ratio:   float = 0.5,   # new
                 cache_presence:   bool  = True):
        self.split = split.lower()
        self.transform = transform
        self.positive_ratio = positive_ratio if self.split == "train" else 0.0
        self.cache_presence = cache_presence and self.split == "train"

        list_file = 'train.txt' if self.split == 'train' else 'test_vol.txt'
        with open(os.path.join(list_dir, list_file)) as f:
            self.sample_list = [ln.strip() for ln in f]

        self.data_dir = os.path.join(base_dir,
            'train_processed_224' if self.split == 'train' else 'val_processed_224')
        
        if self.cache_presence:
            self._is_pos = []
            for name in self.sample_list:
                _, lbl = self._load_npz(name)
                self._is_pos.append(bool(lbl.max()))  # True if any vessel

    def __len__(self):
        return len(self.sample_list)

    def _load_npz(self, slice_name: str):
        fname = slice_name if slice_name.endswith('.npz') else slice_name + '.npz'
        path  = os.path.join(self.data_dir, fname)
        data = np.load(path)
        return data['image'], data['label']

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        image, label = self._load_npz(slice_name)
        sample = {'image': image, 'label': label, 'case_name': slice_name}
        return self.transform(sample) if self.transform else sample
