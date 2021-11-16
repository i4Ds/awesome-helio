# Machine Learning Data Set for NASA's Solar Dynamics Observatory <!-- omit in toc -->

Includes a curated data set from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine-learning research for the period of 2011-01 through 2018-12. 

>In this paper, we present a curated data set from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine-learning research. Beginning from level 1 scientific products we have processed various instrumental corrections, down-sampled to manageable spatial and temporal resolutions, and synchronized observations spatially and temporally. We illustrate the use of this data set with two example applications: forecasting future extreme ultraviolet (EUV) Variability Experiment (EVE) irradiance from present EVE irradiance and translating Helioseismic and Magnetic Imager observations into Atmospheric Imaging Assembly observations. For each application, we provide metrics and baselines for future model comparison. We anticipate this curated data set will facilitate machine-learning research in heliophysics and the physical sciences generally, increasing the scientific return of the SDO mission. This work is a direct result of the 2018 NASA Frontier Development Laboratory Program. Please see the Appendix for access to the data set, totaling 6.5TBs.

the total size is 6.5TB

- [Data Access](#data-access)
  - [Script](#script)
  - [NAS](#nas)
- [Cite](#cite)

## Data Access

The dataset can be downloaded from the [Stanford Digital Repository](https://purl.stanford.edu/nk828sc2920)

### Script

Alternatively, the following script can be used

```
./download.sh 94 2012 ./data/sdo 2018
```

on `macos` use

```
brew install bash wget
/usr/local/bin/bash ./download.sh 94 2012 ./data/sdo 2018
```

### NAS

The full dataset can be found on the FHNW NAS under data02/sdo/machine-learning-dataset.

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