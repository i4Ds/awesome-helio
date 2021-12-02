# SDOBenchmark dataset <!-- omit in toc -->

Image dataset extracted from the Solar Dynamics Observatory (SDO) mission’s AIA and HMI instruments computed for the period of 2012-01 through 2018-01, for 10 wavelength channels. 

>This dataset provides multiple samples of rendered SDO AIA and HMI FITS images of NOAA active regions, alongside a GOES X-ray peak flux which has to be predicted. The active region images are created at 4 time steps before the peak flux prediction window (12h, 5h, 1.5h and 10min). The peak flux prediction windows are the next 24 hours. 
>
>The data set is split into training and test samples. A single sample consists of 4 time steps, being each 1 hour apart, and a target peak flux.
>
>A time step consists of multiple 256x256 image patches. Each image patch corresponds to a single AIA wavelength or HMI image. If images for all wavelengths are available, a single time step thus contains 10 image patches, less otherwise. All images of a sample are within one folder, therefore containing up to 40 images.
>
>The training and test sets are sampled in a way which prevents training set bias to influence test set results: Active regions in the training and test sets are mutually exclusive.

the total size is 4G GB

-- [Data Access](#data-access)
  - [NAS](#nas)
- [Cite](#cite)

## Data Access

The dataset can be downloaded from the [SDOBenchmark webpage](https://i4ds.github.io/SDOBenchmark/)


### NAS

Dataset can be found on the FHNW NAS under data02/sdo/sdobenchmark-dataset.

## Cite

```
Algorithm used to create the dataset from the raw data: https://github.com/i4Ds/SDOBenchmark/blob/master/STRUCTURE.md 
```