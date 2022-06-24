# Data @ i4DS

## Datasets

There are two partitions on NAS05 dedicated to astro ML datasets `astrodata01` and `astrodata02`.

### astrodata01 

Different datasets used for AstroML.

- [goes/goes_ts_2010-2020.parquet.zip](https://www.ngdc.noaa.gov/stp/satellite/goes-r.html) - GOES timeseries downloaded with `sdo-cli` in Parquet format. Refer to [this notebook](https://github.com/i4Ds/sdo-cli/blob/main/notebooks/GOES.ipynb) for an example. This dataset can be used as GOES cache for the [SDO ML v2 data loader](https://github.com/i4Ds/sdo-cli/blob/main/src/sdo/sood/data/sdo_ml_v2_dataset.py). 
- [sdo_benchmark](https://github.com/i4Ds/SDOBenchmark) - Downloaded version of SDOBenchmark.
- [sdo_raw]() - ??
- [sdomlv1_full](https://purl.stanford.edu/nk828sc2920) -  Full version of the SDO ML v1 dataset. [Code example](https://gitlab.com/jdonzallaz/solarnet/-/tree/master/)  [SDO ML v1 data loader](https://github.com/i4Ds/sdo-cli/blob/main/src/sdo/sood/data/sdo_ml_v1_dataset.py)
- [sdomlv1_mini](https://github.com/dfouhey/sdodemo) - Substantially temporally downsized version of the SDO Dataset of A Machine-learning Data Set Prepared from the NASA Solar Dynamics Observatory Mission.
- [sdomlv2_full](https://sdoml.github.io/#/?id=main) - Full version of the SDO ML v2 dataset. [SDO ML v2 data loader](https://github.com/i4Ds/sdo-cli/blob/main/src/sdo/sood/data/sdo_ml_v2_dataset.py)
- [sdomlv2_small](https://sdoml.github.io/#/?id=main) - Small version of the SDO ML v2 dataset. [SDO ML v2 data loader](https://github.com/i4Ds/sdo-cli/blob/main/src/sdo/sood/data/sdo_ml_v2_dataset.py)
- [sharp/derived_sharp_dataset](https://gitlab.fhnw.ch/sharp-flare-forecasting) - A derived SHARP dataset used for investigating handcrafted features and automatic features extracted with a VAE. [SHARP Flare Forecasting](https://gitlab.fhnw.ch/sharp-flare-forecasting) 
- [swan_sf](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM) Downloaded version of multivariate time series (MVTS) data extracted from HMI Active Region Patch (SHARP) series.

```
/mnt/nas05/astrodata01/astroml_data
├── README.md
├── curated_image_parameter_dataset
├── goes
│   └── goes_ts_2010-2020.parquet.zip
├── sdo_benchmark
├── sdo_raw
│   ├── aiadata
│   └── hmidata
├── sdomlv1_full
│   ├── stanford_machine_learning_dataset_for_sdo
│   └── stanford_machine_learning_dataset_for_sdo_extracted
├── sdomlv1_mini
│   ├── 2011.tar
│   ├── 2012.tar
│   ├── 2013.tar
│   ├── 2014.tar
│   ├── 2015.tar
│   ├── 2016.tar
│   ├── 2017.tar
│   ├── 2018.tar
│   └── EVEPlusMeta.tar
├── sdomlv2_full
│   ├── sdomlv2.zarr
│   ├── sdomlv2_eve.zarr
│   └── sdomlv2_hmi.zarr
├── sdomlv2_small
│   ├── sdomlv2_hmi_small.zarr
│   └── sdomlv2_small.zarr
├── sharp
│   ├── derived_sharp_dataset
└── swan_sf
```

### astrodata02 

IRIS datasets

- [iris](https://github.com/i4Ds/IRISdataset/tree/master) - Dataset with selected representative observations for different observation classes.

```
/mnt/nas05/astrodata02
├── iris
│   ├── 2013
│   ├── 2014
│   ├── 2015
│   ├── 2016
│   ├── 2017
│   ├── 2018
│   ├── find_empty_obs
│   ├── find_stray_zips
│   ├── precompute_valid_steps
│   ├── set_permissions
│   └── uncompress_all
└── only-for-iris-data
```


