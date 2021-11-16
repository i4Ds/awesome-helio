# Machine Learning Data Set for NASA's Solar Dynamics Observatory <!-- omit in toc -->

Includes a curated data set from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine-learning research for the period of 2011-01 through 2018-12. 

>In this paper, we present a curated data set from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine-learning research. Beginning from level 1 scientific products we have processed various instrumental corrections, down-sampled to manageable spatial and temporal resolutions, and synchronized observations spatially and temporally. We illustrate the use of this data set with two example applications: forecasting future extreme ultraviolet (EUV) Variability Experiment (EVE) irradiance from present EVE irradiance and translating Helioseismic and Magnetic Imager observations into Atmospheric Imaging Assembly observations. For each application, we provide metrics and baselines for future model comparison. We anticipate this curated data set will facilitate machine-learning research in heliophysics and the physical sciences generally, increasing the scientific return of the SDO mission. This work is a direct result of the 2018 NASA Frontier Development Laboratory Program. Please see the Appendix for access to the data set, totaling 6.5TBs.

the total size is 6.5TB

- [Data Access](#data-access)
  - [Download using a script](#download-using-a-script)
  - [Loading the data using a script](#loading-the-data-using-a-script)
  - [NAS](#nas)
- [Cite](#cite)

## Data Access

The dataset can be downloaded from the [Stanford Digital Repository](https://purl.stanford.edu/nk828sc2920)

### Download using a script

__Credit: Jonathan Donzallaz__

Alternatively, the following script can be used

```
./download.sh 94 2012 ./data/sdo 2018
```

on `macos` use

```
brew install bash wget
/usr/local/bin/bash ./download.sh 94 2012 ./data/sdo 2018
```

### Loading the data using a script

The data can the be extracted as follows:

```
./extract.sh Bz 2013 ./data 2018
```

this will result in the following folder structure

```
datadir/
└───<instrument>/
│   └───<year>/
│       └───<month>/
│           └───<day>/
                │   <instrument><year><month><day>_<hhmm>_<channel>.npz
```

The data is stored as follows:

- AIA: as a set of images of the form ({$year}/AIA/{$wavelength}/{$month}/{$day}/AIA{$year}{$month}{$day}_{$hour}{$minute}_{$wavelength}.npz) e.g., AIA/2013/AIA/0211/11/25/AIA20131125_0824_0211.npz
- HMI : as a set of images of the form ({$year}/HMI/{$wavelength}/{$month}/{$day}/HMI{$year}{$month}{$day}_{$hour}{$minute}_{$wavelength}.npz) e.g., HMI/2013/HMI/bz/11/25/HMI20131125_0824_bz.npz
- EVE: as a single numpy file EVE/irradiance.npy, where each row is a date in time. Any invalid datapoint is set as -1, which you should specially handle or delete. The ones primarily of interest are from MEGS-A: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14] (note the missing 13); Indices 15 and higher are rarely observed.

Each image is stored as a variable 'x' in each npz. For compression reasons, the data is stored as float16 if the data fits and float32 otherwise. You should immediately, however, convert the data to at least float32.

The instruments' data are joined by join.csv, which has the form

```
eve_ind,reduced_ind,time,94,131,171,193,211,304,335,1600,1700,bx,by,bz
...
22154,1874364,2013-11-25 08:24:00,2013/AIA/0094/11/25/AIA20131125_0824_0094.npz,2013/AIA/0131/11/25/AIA20131125_0824_0131.npz,2013/AIA/0171/11/25/AIA20131125_0824_0171.npz,2013/AIA/0193/11/25/AIA20131125_0824_0193.npz,2013/AIA/0211/11/25/AIA20131125_0824_0211.npz,2013/AIA/0304/11/25/AIA20131125_0824_0304.npz,2013/AIA/0335/11/25/AIA20131125_0824_0335.npz,2013/AIA/1600/11/25/AIA20131125_0824_1600.npz,2013/AIA/1700/11/25/AIA20131125_0824_1700.npz,2013/HMI/bx/11/25/HMI20131125_0824_bx.npz,2013/HMI/by/11/25/HMI20131125_0824_by.npz,2013/HMI/bz/11/25/HMI20131125_0824_bz.npz
...
```

Each line corresponds to a data point:

- eve_ind: the index into irradiance.npy, or None if the data point is not valid
- reduce_ind: the index into the original irradiance.npy file
- time: the time of the observation
- (94/131/171/193/211/304/335/1600/1700): the corresponding AIA files
- (bx/by/bz): the corresponding HMI files

### NAS

The full dataset can be found on the FHNW NAS under `data02/sdo/stanford_machine_learning_dataset_for_sdo`.

## Cite

```
@article{galvez2019machine,
  title={A machine-learning data set prepared from the NASA solar dynamics observatory mission},
  author={Galvez, Richard and Fouhey, David F and Jin, Meng and Szenicer, Alexandre and Mu{\~n}oz-Jaramillo, Andr{\'e}s and Cheung, Mark CM and Wright, Paul J and Bobra, Monica G and Liu, Yang and Mason, James and others},
  journal={The Astrophysical Journal Supplement Series},
  volume={242},
  number={1},
  pages={7},
  year={2019},
  publisher={IOP Publishing}
}
```