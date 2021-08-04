# Awesome Helio  <!-- omit in toc -->

A curated list of datasets, tools and papers for machine learning in heliophysics.

## About

The primary goal of `awesome-helio` is to provide a compiled list for all the great resources contributing to the state of the art in heliophysics that are out there but are often hard to find. `awesome-helio` is mainly focussed on SDO and IRIS data and targets machine learning groups but can also provide important insights for other users.

## Contributing

Please take a quick look at the [contribution guidelines](https://github.com/i4Ds/awesome-helio/blob/master/CONTRIBUTING.md) first.

**_If you find any datasets, tools and papers that is missing or is not a good fit, please submit a pull request to improve this file. Thank you!_**

### Contents  <!-- omit in toc -->

- [About](#about)
- [Contributing](#contributing)
- [Missions](#missions)
- [Datasets](#datasets)
  - [Data formats](#data-formats)
  - [e-Callisto](#e-callisto)
  - [IRIS](#iris)
  - [RHESSI](#rhessi)
  - [SDO](#sdo)
  - [STIX](#stix)
- [Tools](#tools)
- [Books](#books)
- [Papers](#papers)
  - [Datasets](#datasets-1)
  - [Event detection, classification and tracking](#event-detection-classification-and-tracking)
  - [Missions](#missions-1)
  - [Space weather prediction](#space-weather-prediction)
  - [Other Applications](#other-applications)
- [Videos](#videos)

## Missions

SDO and IRIS are part of a larger fleet of spacecraft strategically placed throughout our heliosphere:

<span class="img_container center" style="display: block;">
    <img alt="Heliophysics Mission Fleet" src="../main/images/helio-fleet-01-2021.jpg?raw=true" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Heliophysics Mission Fleet, Source <a href="https://science.nasa.gov/heliophysics/mission-fleet-diagram">NASA</a></span>
</span>
</br>

A comprehensive list of different missions can be found [here](https://science.nasa.gov/missions-page?field_division_tid=5&field_phase_tid=All).

## Datasets

For datasets at FHNW refer to [this link](DATA_at_FHNW.md).

### Data formats

**JPEG vs. Fits (Flexible Image Transport System - commonly used digital file format in astronomy)**

### e-Callisto

- [e-Callisto Data access](http://www.e-callisto.org/Data/data.html) - The CALLISTO spectrometer is a programmable heterodyne receiver used for observation of solar radio bursts for astronomical science, education, outreach and citizen science as well as rfi-monitoring.

### IRIS

- [LMSAL data search](https://iris.lmsal.com/search/) - Useful search interface by Lockheed Martin with links to various resources for each observation. Use our [internal](http://server0090.cs.technik.fhnw.ch/iris/) mirror for data download or work directly on one of the dedicated machines.
- [IRISdataset](https://github.com/i4Ds/IRISdataset/tree/master/irisdataset) (Currently only for internal access) - Dataset with selected representative observations for different observation classes.

### RHESSI

TODO


### SDO 

- [Curated Image Parameter Dataset](http://dmlab.cs.gsu.edu/dmlabapi/isd_docs.html) - Massive image parameter dataset extracted from the Solar Dynamics Observatory (SDO) mission’s AIA instrument, for the period of January 2011 through the current date, with the cadence of six minutes, for nine wavelength channels.
- [JSOC](http://jsoc.stanford.edu/) - Data products from the Solar Dynamics Observatory, as well as certain other missions and instruments, are available from the JSOC database.
- [LSDO](https://dataverse.harvard.edu/dataverse/lsdo) - A Large-scale Solar Dynamics Observatory image dataset for computer vision applications Dataverse.
- [Machine Learning Data Set for NASA's Solar Dynamics Observatory](https://purl.stanford.edu/nk828sc2920) - A curated dataset from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine learning research.
- [SDOBenchmark](https://i4ds.github.io/SDOBenchmark/) - Machine learning image dataset for the prediction of solar flares.
- [SOHO/MDI & SDO/HMI line-of-sight Magnetogram Dataset](http://spaceml.org/repo/project/605326665a5e160011fe1175/true) - Curated dataset consisting of co-aligned, co-temporal observations of the same physical structures as observed by HMI and MDI (rotated to the corresponding the HMI frame), ideal for learning-based super-resolution techniques.
- [Solar Flare Prediction from Time Series of Solar Magnetic Field Parameters](https://www.kaggle.com/c/bigdata2019-flare-prediction/data) - Processed dataset provided for the IEEE Big Data 2019 Big Data Cup consisting of a set of magnetic field parameters calculated from individual SHARPs.
- [SWAN-SF](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM) Multivariate time series (MVTS) data extracted from HMI Active Region Patch (SHARP) series.

### STIX

- [STIX Data Center](https://pub023.cs.technik.fhnw.ch/view/list/bsd) - Spectrometer Telescope for Imaging X-rays (STIX) on Solar Orbiter is a hard X-ray imaging spectrometer covering the energy range from 4 to 150 keV

## Tools

- [Aiapy](https://gitlab.com/LMSAL_HUB/aia_hub/aiapy) - Package for analyzing calibrated (level 1) EUV imaging data from AIA.
- [IRISreader](https://github.com/i4Ds/IRISreader) - Python library that allows for efficient browsing through IRIS satellite data in order to simplify machine learning applications.
- [Global High-Resolution Hα network](http://www.bbso.njit.edu/Research/Halpha/)
- [Helioviewer](http://www.helioviewer.org) - Solar and heliospheric image visualization tool.
- [SolarMonitor](https://solarmonitor.org/) - Provides realtime information about the solar activity. The most recent images of the sun are shown, together with the description of the different NOAA AR of the day and which flares has been associated to them.
- [SpaceML](http://spaceml.org/repo) - A Machine Learning toolbox and developer community building the next generation AI applications for space science and exploration containing a set of [code examples](https://github.com/spaceml-org/helionb-sdoml).
- [Sunpy](https://sunpy.org/) - Set of tools for solar data analysis with Python.

## Books

- [Machine Learning, Statistics, and Data Mining for Heliophysics](https://helioml.github.io/HelioML) - Bobra, Monica, and James Paul Mason. "Machine learning, statistics, and data mining for heliophysics."

## Papers


### Datasets


- Kucuk, Ahmet, Juan M. Banda, and Rafal A. Angryk. "A large-scale solar dynamics observatory image dataset for computer vision applications." Scientific data 4.1 (2017): 1-9. [link](https://www.nature.com/articles/sdata201796)
- Ahmadzadeh, Azim, Dustin J. Kempton, and Rafal A. Angryk. "A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission." The Astrophysical Journal Supplement Series 243.1 (2019): 18. [link](https://arxiv.org/abs/1906.01062)
- Galvez, Richard, et al. "A machine-learning data set prepared from the NASA solar dynamics observatory mission." The Astrophysical Journal Supplement Series 242.1 (2019): 7. [link](https://arxiv.org/abs/1903.04538)
- Angryk, Rafal A., et al. "Multivariate time series dataset for space weather data analytics." Scientific data 7.1 (2020): 1-13. [link](https://www.nature.com/articles/s41597-020-0548-x.pdf) [code](https://bitbucket.org/gsudmlab/workspace/projects/FP)


### Event detection, classification and tracking

- Schuh, Michael A., Dustin Kempton, and Rafal A. Angryk. "A Region-Based Retrieval System for Heliophysics Imagery." FLAIRS Conference. 2017. [link](https://www.researchgate.net/profile/Dustin-Kempton-2/publication/317385457_A_Region-based_Retrieval_System_for_Heliophysics_Imagery/links/5cf7a11692851c4dd02a3da9/A-Region-based-Retrieval-System-for-Heliophysics-Imagery.pdf)
- Kucuk, Ahmet, Juan M. Banda, and Rafal A. Angryk. "Solar event classification using deep convolutional neural networks." International Conference on Artificial Intelligence and Soft Computing. Springer, Cham, 2017. [link](https://www.researchgate.net/publication/317570870)
- Illarionov, Egor A., and Andrey G. Tlatov. "Segmentation of coronal holes in solar disc images with a convolutional neural network." Monthly Notices of the Royal Astronomical Society 481.4 (2018): 5014-5021. [link](https://arxiv.org/abs/1809.05748)
- Kempton, Dustin J., Michael A. Schuh, and Rafal A. Angryk. "Tracking solar phenomena from the sdo." The Astrophysical Journal 869.1 (2018): 54. [link](https://iopscience.iop.org/article/10.3847/1538-4357/aae9e9)
- Armstrong, John A., and Lyndsay Fletcher. "Fast solar image classification using deep learning and its importance for automation in solar physics." Solar Physics 294.6 (2019): 1-23. [link](https://link.springer.com/article/10.1007/s11207-019-1473-z)
- Love, Teri, Thomas Neukirch, and Clare E. Parnell. "Analyzing AIA Flare Observations Using Convolutional Neural Networks." Frontiers in Astronomy and Space Sciences 7 (2020): 34. [link](https://doi.org/10.3389/fspas.2020.00034)

### Missions

- Pesnell, W. Dean, B. J. Thompson, and P. C. Chamberlin. "The solar dynamics observatory (SDO)." The Solar Dynamics Observatory. Springer, New York, NY, 2011. 3-15. [link](https://www.researchgate.net/profile/William-Pesnell/publication/236026766_The_Solar_Dynamics_Observatory/links/0c9605287e3a908b99000000/The-Solar-Dynamics-Observatory.pdf)
- Lemen, James R., et al. "The atmospheric imaging assembly (AIA) on the solar dynamics observatory (SDO)." The solar dynamics observatory. Springer, New York, NY, 2011. 17-40. [link](https://link.springer.com/article/10.1007/s11207-011-9776-8)

### Space weather prediction

- Bobra, Monica G., and Sebastien Couvidat. "Solar flare prediction using SDO/HMI vector magnetic field data with a machine-learning algorithm." The Astrophysical Journal 798.2 (2015): 135. [link](https://iopscience.iop.org/article/10.1088/0004-637X/798/2/135/pdf)
- McGregor, Sean, et al. "Flarenet: A deep learning framework for solar phenomena prediction." Neural Information Processing Systems (NIPS) 2017 workshop on Deep Learning for Physical Sciences (DLPS), Long Beach, CA, US. 2017. [link](http://solardynamo.org/publications/McGregor_etal_NIPS_2017.pdf)
- Nagem, Tarek AM, et al. "Deep learning technology for predicting solar flares from (Geostationary Operational Environmental Satellite) data." (2018) [link](https://www.researchgate.net/publication/322924477_Deep_Learning_Technology_for_Predicting_Solar_Flares_from_Geostationary_Operational_Environmental_Satellite_Data?enrichId=rgreq-c8121ef3caa7c31906fde5bb9c53e014-XXX&enrichSource=Y292ZXJQYWdlOzMyMjkyNDQ3NztBUzo1OTAyMDE3ODA5NzM1NjhAMTUxNzcyNjQ3ODAyNA==&el=1_x_2&_esc=publicationCoverPdf)
- Panos, Brandon, et al. "Identifying typical Mg II flare spectra using machine learning." The Astrophysical Journal 861.1 (2018): 62. [link](https://iopscience.iop.org/article/10.3847/1538-4357/aac779/pdf)
- Jonas, Eric, et al. "Flare prediction using photospheric and coronal image data." Solar Physics 293.3 (2018): 1-22. [link](https://link.springer.com/article/10.1007/s11207-018-1258-9)
- Chen, Yang, et al. "Identifying solar flare precursors using time series of SDO/HMI Images and SHARP Parameters." Space Weather 17.10 (2019): 1404-1426. [link](https://arxiv.org/pdf/1904.00125.pdf)
- Panos, Brandon, and Lucia Kleint. "Real-time flare prediction based on distinctions between flaring and non-flaring active region spectra." The Astrophysical Journal 891.1 (2020): 17. [link](https://iopscience.iop.org/article/10.3847/1538-4357/ab700b/pdf)
- Ivanov, Sergey, et al. "Solar activity classification based on Mg II spectra: towards classification on compressed data." arXiv preprint arXiv:2009.07156 (2020). [link](https://arxiv.org/pdf/2009.07156.pdf)
- Wang, Jingjing, et al. "Solar Flare Predictive Features Derived from Polarity Inversion Line Masks in Active Regions Using an Unsupervised Machine Learning Algorithm." The Astrophysical Journal 892.2 (2020): 140. [link](https://iopscience.iop.org/article/10.3847/1538-4357/ab7b6c/pdf)
- Ahmadzadeh, Azim, et al. "How to Train Your Flare Prediction Model: Revisiting Robust Sampling of Rare Events." The Astrophysical Journal Supplement Series 254.2 (2021): 23. [link](https://iopscience.iop.org/article/10.3847/1538-4365/abec88/pdf)


### Other Applications

- Wright, Paul J., et al. "DeepEM: Demonstrating a Deep Learning Approach to DEM Inversion." Zenodo. [link](https://zenodo.org/record/2587015#.YMjHtZMzau8)
- Guedes dos Santos, L. F., et al. "Multi-Channel Auto-Calibration for the Atmospheric Imaging Assembly instrument with Deep Learning." AGU Fall Meeting Abstracts. Vol. 2020. 2020. [link](https://arxiv.org/abs/2012.14023)

## Videos

- [SDO 2021 Science Workshop](https://www.youtube.com/channel/UCy_NKqf3CnLGlBTBJ5DmkBA) - Recent science topics targeting SDO data
- [Big data 2020 Conference Talk](https://dmlab.cs.gsu.edu/bigdata/bigdata-tutorial-2020/BigData2020_Tutorial_backupTalks2.mp4) - Tutorial 6: Data Sources, Tools, and Techniques for Big Data-driven Machine Learning in Heliophysics
