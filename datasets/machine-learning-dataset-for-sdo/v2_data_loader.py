import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, Normalize, Lambda
import math
import pytorch_lightning as pl
import gcsfs
import zarr
import pandas as pd
import dask.array as da


class SDOMLv2NumpyDataset(Dataset):
    def __init__(
            self,
            storage_root="fdl-sdoml-v2/sdomlv2_small.zarr/",
            n_items=None,
            year="2010",
            channel="171A",
            start="2010-08-28 00:00:00",
            end="2010-08-28 23:59:59",
            freq="6T",
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
        # TODO only load required channels
        gcs = gcsfs.GCSFileSystem(access="read_only")
        store = gcsfs.GCSMap(storage_root, gcs=gcs, check=False)
        root = zarr.group(store)
        print("discovered the following zarr directory structure")
        print(root.tree())

        if year:
            by_year = root[year]
        else:
            by_year = root[:]

        if channel:
            data = by_year[channel]
        else:
            data = by_year[:]

        if freq:
            # temporal downsampling
            t_obs = np.array(data.attrs["T_OBS"])
            df_time = pd.DataFrame(t_obs, index=np.arange(
                np.shape(t_obs)[0]), columns=["Time"])
            df_time["Time"] = pd.to_datetime(df_time["Time"])
            # select times at a frequency of freq (e.g. 12T)
            selected_times = pd.date_range(
                start=start, end=end, freq=freq, tz="UTC"
            )
            selected_index = []
            for i in selected_times:
                selected_index.append(np.argmin(abs(df_time["Time"] - i)))
            time_index = [x for x in selected_index if x > 0]
            all_images = da.from_array(data)[time_index, :, :]
            attrs = {keys: [values[idx] for idx in time_index]
                     for keys, values in data.attrs.items()}
        else:
            attrs = data.attrs
            all_images = da.from_array(data)

        self.transforms = transforms
        self.all_images = all_images
        self.attrs = attrs

        self.data_len = len(self.all_images)
        print(f"found {len(self.all_images)} images")
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
        image = self.all_images[idx, :, :]

        attrs = dict([(key, self.attrs[key][idx])
                     for key in self.attrs.keys()])
        torch_arr = torch.from_numpy(np.array(image))
        # convert to 1 x H x W, to be in compatible torchvision format
        torch_arr = torch_arr.unsqueeze(dim=0)
        if self.transforms is not None:
            torch_arr = self.transforms(torch_arr)

        return torch_arr, attrs


# TODO are these values still applicable for the new correction factors?
# Same preprocess as github.com/i4Ds/SDOBenchmark
CHANNEL_PREPROCESS = {
    "94A": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131A": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171A": {"min": 5, "max": 3500, "scaling": "log10"},
    "193A": {"min": 20, "max": 5500, "scaling": "log10"},
    "211A": {"min": 7, "max": 3500, "scaling": "log10"},
    "304A": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335A": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600A": {"min": 10, "max": 800, "scaling": "log10"},
    "1700A": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500A": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "bx": {"min": -250, "max": 250, "scaling": None},
    "by": {"min": -250, "max": 250, "scaling": None},
    "bz": {"min": -250, "max": 250, "scaling": None},
}


def get_default_transforms(target_size=128, channel="171"):
    """Returns a Transform which resizes 2D samples (1xHxW) to a target_size (1 x target_size x target_size)
    and then converts them to a pytorch tensor.
    Args:
        target_size (int, optional): [New spatial dimension of the input data]. Defaults to 128.
        channel (str, optional): [The SDO channel]. Defaults to 171.
    Returns:
        [Transform]
    """

    """
    Apply the normalization necessary for the sdo-dataset. Depending on the channel, it:
      - flips the image vertically
      - clips the "pixels" data in the predefined range (see above)
      - applies a log10() on the data
      - normalizes the data to the [0, 1] range
      - normalizes the data around 0 (standard scaling)

    :param channel: The kind of data to preprocess
    :param resize: Optional size of image (integer) to resize the image
    :return: a transforms object to preprocess tensors
    """

    # also refer to https://pytorch.org/vision/stable/transforms.html
    # and https://gitlab.com/jdonzallaz/solarnet-thesis/-/blob/master/solarnet/data/transforms.py
    preprocess_config = CHANNEL_PREPROCESS[channel]

    if preprocess_config["scaling"] == "log10":
        # TODO why was vflip(x) used here in SolarNet?
        def lambda_transform(x): return torch.log10(torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        ))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - \
            math.log10(preprocess_config["min"])
    else:
        def lambda_transform(x): return torch.clamp(
            x,
            min=preprocess_config["min"],
            max=preprocess_config["max"],
        )
        mean = preprocess_config["min"]
        std = preprocess_config["max"] - preprocess_config["min"]

    transforms = Compose(
        [Resize((target_size, target_size)),
         Lambda(lambda_transform),
         Normalize(mean=[mean], std=[std]),
         # required to remove strange distribution of pixels (everything too bright)
         Normalize(mean=(0.5), std=(0.5))
         ]
    )
    return transforms


class SDOMLv2DataModule(pl.LightningDataModule):
    def __init__(self,
                 storage_root: str = "fdl-sdoml-v2/sdomlv2_small.zarr/",
                 batch_size: int = 16,
                 n_items: int = None,
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 drop_last: bool = False,
                 target_size: int = 256,
                 channel: str = "171A",
                 year="2010",
                 start="2010-08-28 00:00:00",
                 end="2010-08-28 23:59:59",
                 freq="6T",
                 shuffle: bool = False):
        """
        Creates a LightningDataModule for the SDO Ml v2 dataset

        The header information can be retrieved as the second argument when enumerating the loader

        >>> loader = SDOMLv2DataModule(channel="171A").train_dataloader()
        >>> for batch_idx, batch in enumerate(loader):
        >>>     X, headers  = batch

        Args:
            storage_root ([str]): [Root path in the GCS bucket containing the zarr archives.]
            batch_size (int, optional): [See pytorch DataLoader]. Defaults to 16.
            n_items ([int], optional): [Number of items in the dataset, by default number of files in the loaded set 
                                            but can be smaller (uses subset) or larger (uses file multiple times)]. Defaults to None.
            pin_memory (bool, optional): [See pytorch DataLoader]. Defaults to False.
            num_workers (int, optional): [See pytorch DataLoader]. Defaults to 0.
            drop_last (bool, optional): [See pytorch DataLoader]. Defaults to False.
            target_size (int, optional): [New spatial dimension of to which the input data will be transformed]. Defaults to 256.
            channel (str, optional): [Channel name that should be used]. Defaults to "171A".
            year (str, optional): [Allows to prefilter the dataset by year, useful for train/test splits]. Defaults to "2010".
            start (str, optional): [Allows to restrict the dataset temporally, only works in combination with freq]. Defaults to "2010-08-28 00:00:00".
            end (str, optional): [Allows to restrict the dataset temporally, only works in combination with freq]. Defaults to "2010-08-28 23:59:59".
            freq (str, optional): [Allows to downsample the dataset temporally, should be bigger than the min interval for the observed channel]. Defaults to "6T".
            shuffle (bool, optional): [See pytorch DataLoader]. Defaults to False.
        """
        super().__init__()

        transforms = get_default_transforms(
            target_size=target_size, channel=channel)

        dataset = SDOMLv2NumpyDataset(
            storage_root=storage_root,
            year=year,
            start=start,
            end=end,
            freq=freq,
            n_items=n_items,
            channel=channel,
            transforms=transforms,
        )

        self.dataset_test = SDOMLv2NumpyDataset(
            storage_root=storage_root,
            year=year,
            start=start,
            end=end,
            freq=freq,
            n_items=n_items,
            channel=channel,
            transforms=transforms,
        )

        # TODO investigate the use of a ChunkSampler in order to improve data loading performance https://gist.github.com/wassname/8ae1f64389c2aaceeb84fcd34c3651c3
        # TODO implement temporal split
        num_samples = len(dataset)
        splits = [int(math.floor(num_samples*0.8)),
                  int(math.ceil(num_samples * 0.2))]
        print(f"splitting datatset with {num_samples} into {splits}")
        self.dataset_train, self.dataset_val = random_split(dataset, splits)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)

    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=self.drop_last,)
