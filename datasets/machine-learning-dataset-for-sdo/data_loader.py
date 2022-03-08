import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize


class NumpyDataset(Dataset):
    def __init__(
            self,
            base_dir,
            mode="train",
            n_items=None,
            file_pattern="*.npz",
            data_key="x",
            transforms=None):
        """Dataset which loads Numpy npz files
        Args:
            base_dir ([str]): [Directory in which the npz files are.]
            mode (str, optional): [train or val, TODO implement val split]. Defaults to "train".
            n_items ([type], optional): [Number of items in on iteration, by default number of files in the loaded set 
                                        but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
            file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*.npz".
            data_key (str, optional): [Data key used to load data from the npz array]. Defaults to 'x'.
            transforms ([type], optional): [Transformations to do after loading the data -> pytorch data transforms]. Defaults to None
        """

        self.base_dir = base_dir
        self.file_pattern = file_pattern
        self.files = list(Path(base_dir).glob(f'*/{file_pattern}'))
        self.transforms = transforms
        self.data_key = data_key

        self.data_len = len(self.files)
        if n_items is None:
            self.n_items = self.data_len
        else:
            self.n_items = int(n_items)

    def __len__(self):
        return self.n_items

    def __getitem__(self, item):

        if item >= self.n_items:
            raise StopIteration()

        idx = item % self.data_len
        file_path = str(self.files[idx])
        np_arr = np.load(file_path)[self.data_key]
        torch_arr = torch.from_numpy(np_arr)
        # convert to 1 x H x W, to be in compatible torchvision format
        torch_arr = torch_arr.unsqueeze(dim=0)
        if self.transforms is not None:
            torch_arr = self.transforms(torch_arr)

        return torch_arr, 0


def get_default_transforms(target_size=128):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.
    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 128.
    Returns:
        [Transform]
    """

    #also refer to https://pytorch.org/vision/stable/transforms.html#
    transforms = Compose([Resize((target_size, target_size))])
    return transforms


def get_dataset(
    base_dir,
    mode="train",
    batch_size=16,
    n_items=None,
    pin_memory=False,
    num_processes=1,
    drop_last=False,
    target_size=512,
    file_pattern="*.npz",
    do_reshuffle=True,
):
    """Returns a Pytorch DataLoader which loads a NumpyDataset
    Args:
        base_dir ([str]): [Directory in which the npz files are.]
        mode (str, optional): [train or val, TODO implement train val split]. Defaults to "train".
        batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
        n_items ([int], optional): [Number of items in the dataset, by default number of files in the loaded set 
                                        but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
        pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
        num_processes (int, optional): [See pytorch DataLoader]. Defaults to 1.
        drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
        target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 512.
        file_pattern (str, optional): [File pattern of files to load from the base_dir]. Defaults to "*.npz".
        do_reshuffle (bool, optional): [See pytorch DataLoader]. Defaults to True.
    Returns:
        [DataLoader]: Pytorch data loader which loads a NumpyDataset
    """

    transforms = get_default_transforms(target_size=target_size)

    dataset = NumpyDataset(
        base_dir=base_dir,
        mode=mode,
        n_items=n_items,
        file_pattern=file_pattern,
        transforms=transforms,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_reshuffle,
        num_workers=num_processes,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader
