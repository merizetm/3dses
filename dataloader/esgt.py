import os
import numpy as np
import torch
from torch.utils.data import Dataset


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type="fnv", mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == "ravel":
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count

    """
    #_, idx = np.unique(key, return_index=True)
    #return idx

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, idx_start, count = np.unique(key_sort, return_counts=True, return_index=True)
    idx_list = np.split(idx_sort, idx_start[1:])
    return idx_list"""

def voxelize_and_inverse(coord, hash_type="fnv", mode=0):
    if hash_type == "ravel":
        key = ravel_hash_vec(coord)
    else:
        key = fnv_hash_vec(coord)
    idx_sort = np.argsort(key)
    N = coord.shape[0]
    inv_sort = np.zeros(N, dtype=np.int32)
    inv_sort[idx_sort] = np.arange(N)
    key_sort = key[idx_sort]
    assert np.array_equal(key_sort[inv_sort], key)
    uniq_key, _, inv_idx, counts = np.unique(
        key_sort, return_index=True, return_inverse=True, return_counts=True
    )
    # select the first
    idx_select = np.cumsum(np.insert(counts, 0, 0)[0:-1])
    uniq_idx = idx_sort[idx_select]
    inv_idx = inv_idx[inv_sort]

    assert len(uniq_idx) == len(uniq_key)
    assert np.array_equal(key[uniq_idx], uniq_key)
    assert len(inv_idx), len(key)
    assert np.array_equal(uniq_key[inv_idx], key)
    return uniq_idx, inv_idx


def data_prepare_3dses(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        color = feat[:, 0:3]
        coord, color = transform(coord, color)
        feat[:, 0:3] = color
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        coord = coord / voxel_size
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = (
            np.random.randint(label.shape[0])
            if "train" in split
            else label.shape[0] // 2
        )
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_3dses_point(
    coord,
    feat,
    label,
    split="train",
    voxel_size=0.04,
    voxel_max=None,
    transform=None,
    shuffle_index=False,
):
    if transform:
        color = feat[:, 0:3]
        coord, color = transform(coord, color)
        feat[:, 0:3] = color
    label_pts = label
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        coord = coord / voxel_size
        int_coord = coord.astype(np.int32)

        unique_map, inverse_map = voxelize_and_inverse(int_coord, voxel_size)
        coord = coord[unique_map]
        feat = feat[unique_map]
        label = label[unique_map]

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label_pts = torch.LongTensor(label_pts)
    label = torch.LongTensor(label)
    inverse_map = torch.LongTensor(inverse_map)
    return coord, feat, label, label_pts, inverse_map


class ESGT(Dataset):
    def __init__(
        self,
        split="train",
        data_root="trainval",
        test_area=[172],
        voxel_size=0.04,
        voxel_max=None,
        transform=None,
        shuffle_index=False,
        loop=1,
        vote_num=1,
        fea_channels=[3,4,5],feat_transform=None, probability=0.2,
    ):
        super().__init__()
        (
            self.split,
            self.voxel_size,
            self.transform,
            self.voxel_max,
            self.shuffle_index,
            self.loop,
            self.fea_channels,self.feat_transform, self.probability,
        ) = (split, voxel_size, transform, voxel_max, shuffle_index, loop, fea_channels,feat_transform,probability)
        data_list = sorted(os.listdir(data_root))

        self.test_area = ['S{}'.format(x) for x in test_area]

        data_list = [item[:-4] for item in data_list if "S" in item]
        if split == "train":
            self.data_list = [
                item for item in data_list if item not in self.test_area
            ]
        else:
            self.data_list = [
                item for item in data_list if item in self.test_area
            ]
        self.data_root = data_root
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        self.data_idx = np.repeat(self.data_idx, vote_num)
        print("Total repeated {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + ".npy")
        data = np.load(data_path)

        if 'intensity' in self.fea_channels : # Load RGB + Intensity
            coord, feat, label = data[:, 0:3], data[:, 3:7], data[:, -1]

        else : # Load only RGB
            coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, -1]

        # Add an option to differency manual and auto labels -!
        if 'manuallabel' in self.fea_channels :
            print('manual label')
            label = data[:, -2]
            
        feat[:, 0:3] = feat[:, 0:3] / 127.5 - 1
        coord, feat, label = data_prepare_3dses(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )
        print(coord, coord.shape) ; print(feat, feat.shape) ; print(label, label.shape)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


class ESGT_Point(ESGT):
    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item = self.data_list[data_idx]
        data_path = os.path.join(self.data_root, item + ".npy")
        data = np.load(data_path)

        if 'intensity' in self.fea_channels : # Load RGB + Intensity
            coord, feat, label = data[:, 0:3], data[:, 3:7], data[:, -1]
        else : # Load only RGB
            coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, -1]

        # Add an option to differency manual and auto labels
        if 'manuallabel' in self.fea_channels :
            label = data[:, -2]

        feat[:, 0:3] = feat[:,0:3] / 127.5 - 1
        coord, feat, label, label_pts, inverse_map = data_prepare_3dses_point(
            coord,
            feat,
            label,
            self.split,
            self.voxel_size,
            self.voxel_max,
            self.transform,
            self.shuffle_index,
        )
        return coord, feat, label, label_pts, inverse_map 
