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
- [Books](#books)
- [Datasets](#datasets)
  - [Data formats](#data-formats)
  - [e-Callisto](#e-callisto)
  - [IRIS](#iris)
  - [RHESSI](#rhessi)
  - [SDO](#sdo)
  - [SOHO](#soho)
  - [STIX](#stix)
- [Links](#links)
- [Papers](#papers)
  - [Datasets](#datasets-1)
  - [Event detection, classification and tracking](#event-detection-classification-and-tracking)
  - [Missions](#missions-1)
  - [Space weather prediction](#space-weather-prediction)
  - [Other Applications](#other-applications)
- [Tools](#tools)
- [Videos](#videos)

## Missions

SDO and IRIS are part of a larger fleet of spacecraft strategically placed throughout our heliosphere:

<span class="img_container center" style="display: block;">
    <img alt="Heliophysics Mission Fleet" src="../main/images/helio-fleet-01-2021.jpg?raw=true" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Heliophysics Mission Fleet, Source <a href="https://science.nasa.gov/heliophysics/mission-fleet-diagram">NASA</a></span>
</span>
</br>

A comprehensive list of different missions can be found [here](https://science.nasa.gov/missions-page?field_division_tid=5&field_phase_tid=All).

## Books

- [Machine Learning, Statistics, and Data Mining for Heliophysics](https://helioml.github.io/HelioML) - Bobra, Monica, and James Paul Mason. "Machine learning, statistics, and data mining for heliophysics."
- [Deep Learning in Solar Astronomy](https://link.springer.com/book/10.1007/978-981-19-2746-1) - Xu, Long, Yihua Yan, and Xin Huang. "Deep Learning in Solar Astronomy."

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

Before working with SDO data make sure to checkout the [Guide to SDO Data Analysis](https://www.lmsal.com/sdodocs/doc/dcur/SDOD0060.zip/zip/entry/index.html), which contains a set of instructions on how to browse, find, download, and analyze SDO data.

- [Curated Image Parameter Dataset](http://dmlab.cs.gsu.edu/dmlabapi/isd_docs.html) - Massive image parameter dataset extracted from the Solar Dynamics Observatory (SDO) mission's AIA instrument, for the period of January 2011 through the current date, with the cadence of six minutes, for nine wavelength channels.
- [JSOC](http://jsoc.stanford.edu/) - Data products from the Solar Dynamics Observatory, as well as certain other missions and instruments, are available from the JSOC database.
- [DeepSDO Event dataset](http://sdo.kasi.re.kr/dataset_deepsdo_event.aspx) - Dataset curated by experts containing three solar event categories (coronal holes, sunspots, and prominences). Suitable for object detection using deep learning-based models. 
- [LSDO](https://dataverse.harvard.edu/dataverse/lsdo) - A Large-scale Solar Dynamics Observatory image dataset for computer vision applications Dataverse.
- [Machine Learning Data Set for NASA's Solar Dynamics Observatory](https://purl.stanford.edu/nk828sc2920) - A curated dataset from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine learning research.
- [Machine Learning Data Set for NASA's Solar Dynamics Observatory - not corrected for degradation](https://zenodo.org/record/4430801#.X_xuPOlKhmE) - Version of the SDO ML v1 dataset not corrected for degradation over time.
- [Machine Learning Data Set for NASA's Solar Dynamics Observatory v2](https://sdoml.github.io/#/?id=main) - The second version of the curated dataset from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine learning research. [colab](https://colab.research.google.com/github/spaceml-org/helionb-sdoml/blob/main/notebooks/01_sdoml_dataset_2018/sdoml_dataset_colab.ipynb#scrollTo=u8yLRK4x30gr) [intro](https://www.youtube.com/watch?v=5KNFmnyWRQk) [code](https://github.com/SDOML/SDOMLv2)
- [SDOBenchmark](https://i4ds.github.io/SDOBenchmark/) - Machine learning image dataset for the prediction of solar flares.
- [SOHO/MDI & SDO/HMI line-of-sight Magnetogram Dataset](http://spaceml.org/repo/project/605326665a5e160011fe1175/true) - Curated dataset consisting of co-aligned, co-temporal observations of the same physical structures as observed by HMI and MDI (rotated to the corresponding the HMI frame), ideal for learning-based super-resolution techniques. [colab](https://colab.research.google.com/drive/1mOP-Zx8NPhtLzb17fxi56EdD_Wn6VFuZ?usp=sharing) [code](https://github.com/spaceml-org/helionb-mag) [intro](https://www.youtube.com/watch?v=rhxyC9MkWGU)
- [SOHO/SDO ML Ready Dataset](https://github.com/cshneider/soho-ml-data-ready) - Code to generate and temporally sync SoHO and/or SDO Mission image products to make a standardized machine-learning-ready dataset.
- [Solar Flare Prediction from Time Series of Solar Magnetic Field Parameters](https://www.kaggle.com/c/bigdata2019-flare-prediction/data) - Processed dataset provided for the IEEE Big Data 2019 Big Data Cup consisting of a set of magnetic field parameters calculated from individual SHARPs.
- [SWAN-SF](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM) Multivariate time series (MVTS) data extracted from HMI Active Region Patch (SHARP) series.

When working with AIA data and analyzing short-lived phenomena, make sure to consider respiking the data as the AIA despiking algorithm might remove some of the features (Young, Peter R., et al. "An Analysis of Spikes in Atmospheric Imaging Assembly (AIA) Data." Solar Physics 296.12 (2021): 1-21.).

### SOHO

- Shneider, Carl, et al. "A Machine-Learning-Ready Dataset Prepared from the Solar and Heliospheric Observatory Mission." arXiv preprint arXiv:2108.06394 (2021).

### STIX

- [STIX Data Center](https://pub023.cs.technik.fhnw.ch/view/list/bsd) - Spectrometer Telescope for Imaging X-rays (STIX) on Solar Orbiter is a hard X-ray imaging spectrometer covering the energy range from 4 to 150 keV

## Links

- [AGU Link Collection](https://connect.agu.org/spa/resources) - Solar physics link collection.
- [Astrophysics Data System (ADS)](https://ui.adsabs.harvard.edu/) - Digital library portal for researchers in astronomy and physics.
- [Carrington rotation date list](http://umtof.umd.edu/pm/crn/)
- [Community Coordinated Modeling Center](https://ccmc.gsfc.nasa.gov/index.php) - Partnership for space science and space weather models.
- [Dr Peter R. Young's website](https://www.pyoung.org/) - Useful materials for different missions. 
- [FHNW - Astroinformatics and Heliophysics](https://astro-helio.ch/) - Space-related research at the FHNW Institute for Data Science.
- [Helionauts](https://helionauts.org/) - Heliophysics Forum.
- [Helioanalytics](https://sites.google.com/view/helioanalytics/resources) - Research group and resources for data analytics in heliophysics.
- [Heliophysics Data Portal](https://heliophysicsdata.gsfc.nasa.gov/websearch/dispatcher)
- [Heliophysics Data Environment](https://hdrl.gsfc.nasa.gov/) - Data and Services for the Heliophysics System Observatory.
- [LMSAL AIA](https://aia.lmsal.com/) - Home of SDO's atmospheric imaging assembly (AIA).
- [SDO Documentation](https://www.lmsal.com/sdodocs/)
- [SDO Guide](https://sdo.gsfc.nasa.gov/assets/docs/SDO_Guide.pdf) - A comprehensive booklet with information about the mission.
- [SDO Mission](https://sdo.gsfc.nasa.gov/mission/)
- [SDO Quick Movie Browser](https://sdo.gsfc.nasa.gov/data/aiahmi/)
- [Spacewheater.com](https://spaceweather.com/) - News and information about the Sun-Earth environment.
- [Spaceweatherlive.com](https://www.spaceweatherlive.com/) - Real-time auroral and solar activity.
- [Space wheater data portal](https://lasp.colorado.edu/space-weather-portal/data) - Tool for visualizing, downloading and co-plotting space wheather data.
- [SSCWeb Satellite Situation Center](https://sscweb.gsfc.nasa.gov/) -  Locations for most helio spacecraft.
- [Sungate](https://www.lmsal.com/sungate/) - Portal to solar data, events, and search tools.

## Papers

### Datasets

- Kucuk, Ahmet, Juan M. Banda, and Rafal A. Angryk. "A large-scale solar dynamics observatory image dataset for computer vision applications." Scientific data 4.1 (2017): 1-9. [link](https://www.nature.com/articles/sdata201796)
- McGranaghan, R., et al. "Machine learning databases used for Journal of Geophysical Research: Space Physics manuscript: New capabilities for prediction of high-latitude ionospheric scintillation: A novel approach with machine learning.". (2018). figshare. Dataset. [link](https://doi.org/10.6084/m9.figshare.6813131.v1) 
- Ahmadzadeh, Azim, Dustin J. Kempton, and Rafal A. Angryk. "A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission." The Astrophysical Journal Supplement Series 243.1 (2019): 18. [link](https://arxiv.org/abs/1906.01062)
- Galvez, Richard, et al. "A machine-learning data set prepared from the NASA solar dynamics observatory mission." The Astrophysical Journal Supplement Series 242.1 (2019): 7. [link](https://arxiv.org/abs/1903.04538)
- Baek, Ji-Hye, et al. "Solar Event Detection Using Deep-Learning-Based Object Detection Methods." Solar Physics 296.11 (2021): 1-15. [link](https://link.springer.com/article/10.1007/s11207-021-01902-5)
- Angryk, Rafal A., et al. "Multivariate time series dataset for space weather data analytics." Scientific data 7.1 (2020): 1-13. [link](https://www.nature.com/articles/s41597-020-0548-x.pdf) [code](https://bitbucket.org/gsudmlab/workspace/projects/FP)
- Mahajan, Sushant S., et al. "Improved Measurements of the Sun's Meridional Flow and Torsional Oscillation from Correlation Tracking on MDI and HMI Magnetograms." The Astrophysical Journal 917.2 (2021): 100. [link](https://arxiv.org/abs/2107.07731) [data](https://dataverse.harvard.edu/dataverse/lct-on-solar-magnetograms)
- Shneider, Carl, et al. "A Machine-Learning-Ready Dataset Prepared from the Solar and Heliospheric Observatory Mission." arXiv preprint arXiv:2108.06394 (2021). [link](https://arxiv.org/abs/2108.06394)
- Bobra, Monica G., et al. "SMARPs and SHARPs: Two Solar Cycles of Active Region Data." The Astrophysical Journal Supplement Series 256.2 (2021): 26. [link](https://iopscience.iop.org/article/10.3847/1538-4365/ac1f1d/pdf) [code](https://github.com/mbobra/SMARPs)

### Event detection, classification and tracking

- Martens, P. C. H., et al. "Computer vision for the solar dynamics observatory (SDO)." Solar Physics 275.1 (2012): 79-113. [link](https://link.springer.com/article/10.1007/s11207-010-9697-y)
- Schuh, Michael A., Dustin Kempton, and Rafal A. Angryk. "A Region-Based Retrieval System for Heliophysics Imagery." FLAIRS Conference. 2017. [link](https://www.researchgate.net/profile/Dustin-Kempton-2/publication/317385457_A_Region-based_Retrieval_System_for_Heliophysics_Imagery/links/5cf7a11692851c4dd02a3da9/A-Region-based-Retrieval-System-for-Heliophysics-Imagery.pdf)
- Kucuk, Ahmet, Juan M. Banda, and Rafal A. Angryk. "Solar event classification using deep convolutional neural networks." International Conference on Artificial Intelligence and Soft Computing. Springer, Cham, 2017. [link](https://www.researchgate.net/publication/317570870)
- Illarionov, Egor A., and Andrey G. Tlatov. "Segmentation of coronal holes in solar disc images with a convolutional neural network." Monthly Notices of the Royal Astronomical Society 481.4 (2018): 5014-5021. [link](https://arxiv.org/abs/1809.05748)
- Kempton, Dustin J., Michael A. Schuh, and Rafal A. Angryk. "Tracking solar phenomena from the sdo." The Astrophysical Journal 869.1 (2018): 54. [link](https://iopscience.iop.org/article/10.3847/1538-4357/aae9e9)
- Armstrong, John A., and Lyndsay Fletcher. "Fast solar image classification using deep learning and its importance for automation in solar physics." Solar Physics 294.6 (2019): 1-23. [link](https://link.springer.com/article/10.1007/s11207-019-1473-z)
- Gitiaux, Xavier, et al. "Probabilistic Super-Resolution of Solar Magnetograms: Generating Many Explanations and Measuring Uncertainties." arXiv preprint arXiv:1911.01486 (2019). [link](https://arxiv.org/pdf/1911.01486.pdf)
- Jungbluth, Anna, et al. "Single-frame super-resolution of solar magnetograms: Investigating physics-based metrics\& losses." arXiv preprint arXiv:1911.01490 (2019). [link](https://arxiv.org/pdf/1911.01490.pdf)
- Love, Teri, Thomas Neukirch, and Clare E. Parnell. "Analyzing AIA Flare Observations Using Convolutional Neural Networks." Frontiers in Astronomy and Space Sciences 7 (2020): 34. [link](https://doi.org/10.3389/fspas.2020.00034)
- Mackovjak, Šimon, et al. "SCSS-Net: solar corona structures segmentation by deep learning." Monthly Notices of the Royal Astronomical Society 508.3 (2021): 3111-3124. [code](https://github.com/space-lab-sk/scss-net) [link](https://arxiv.org/pdf/2109.10834.pdf)
- Broock, Elena García, Tobías Felipe, and A. Asensio Ramos. "Performance of solar far-side active region neural detection." Astronomy & Astrophysics 652 (2021): A132. [link](https://arxiv.org/pdf/2106.09365.pdf)

### Missions

- Pesnell, W. Dean, B. J. Thompson, and P. C. Chamberlin. "The solar dynamics observatory (SDO)." The Solar Dynamics Observatory. Springer, New York, NY, 2011. 3-15. [link](https://www.researchgate.net/profile/William-Pesnell/publication/236026766_The_Solar_Dynamics_Observatory/links/0c9605287e3a908b99000000/The-Solar-Dynamics-Observatory.pdf)
- Lemen, James R., et al. "The atmospheric imaging assembly (AIA) on the solar dynamics observatory (SDO)." The solar dynamics observatory. Springer, New York, NY, 2011. 17-40. [link](https://link.springer.com/article/10.1007/s11207-011-9776-8)

### Space weather prediction

- Bobra, Monica G., and Sebastien Couvidat. "Solar flare prediction using SDO/HMI vector magnetic field data with a machine-learning algorithm." The Astrophysical Journal 798.2 (2015): 135. [link](https://iopscience.iop.org/article/10.1088/0004-637X/798/2/135/pdf)
- McGregor, Sean, et al. "Flarenet: A deep learning framework for solar phenomena prediction." Neural Information Processing Systems (NIPS) 2017 workshop on Deep Learning for Physical Sciences (DLPS), Long Beach, CA, US. 2017. [link](http://solardynamo.org/publications/McGregor_etal_NIPS_2017.pdf)
- Nagem, Tarek AM, et al. "Deep learning technology for predicting solar flares from (Geostationary Operational Environmental Satellite) data." (2018) [link](https://www.researchgate.net/publication/322924477_Deep_Learning_Technology_for_Predicting_Solar_Flares_from_Geostationary_Operational_Environmental_Satellite_Data?enrichId=rgreq-c8121ef3caa7c31906fde5bb9c53e014-XXX&enrichSource=Y292ZXJQYWdlOzMyMjkyNDQ3NztBUzo1OTAyMDE3ODA5NzM1NjhAMTUxNzcyNjQ3ODAyNA==&el=1_x_2&_esc=publicationCoverPdf)
- Panos, Brandon, et al. "Identifying typical Mg II flare spectra using machine learning." The Astrophysical Journal 861.1 (2018): 62. [link](https://iopscience.iop.org/article/10.3847/1538-4357/aac779/pdf)
- Jonas, Eric, et al. "Flare prediction using photospheric and coronal image data." Solar Physics 293.3 (2018): 1-22. [link](https://link.springer.com/article/10.1007/s11207-018-1258-9)
- McGranaghan, Ryan M., et al. "New capabilities for prediction of high‐latitude ionospheric scintillation: A novel approach with machine learning." Space Weather 16.11 (2018): 1817-1846. [link](https://doi.org/10.1029/2018SW002018)
- Chen, Yang, et al. "Identifying solar flare precursors using time series of SDO/HMI Images and SHARP Parameters." Space Weather 17.10 (2019): 1404-1426. [link](https://arxiv.org/pdf/1904.00125.pdf)
- Panos, Brandon, and Lucia Kleint. "Real-time flare prediction based on distinctions between flaring and non-flaring active region spectra." The Astrophysical Journal 891.1 (2020): 17. [link](https://iopscience.iop.org/article/10.3847/1538-4357/ab700b/pdf)
- Ivanov, Sergey, et al. "Solar activity classification based on Mg II spectra: towards classification on compressed data." arXiv preprint arXiv:2009.07156 (2020). [link](https://arxiv.org/pdf/2009.07156.pdf)
- Wang, Jingjing, et al. "Solar Flare Predictive Features Derived from Polarity Inversion Line Masks in Active Regions Using an Unsupervised Machine Learning Algorithm." The Astrophysical Journal 892.2 (2020): 140. [link](https://iopscience.iop.org/article/10.3847/1538-4357/ab7b6c/pdf)
- Ahmadzadeh, Azim, et al. "How to Train Your Flare Prediction Model: Revisiting Robust Sampling of Rare Events." The Astrophysical Journal Supplement Series 254.2 (2021): 23. [link](https://iopscience.iop.org/article/10.3847/1538-4365/abec88/pdf)
- McGranaghan, Ryan M., et al. "Toward a next generation particle precipitation model: Mesoscale prediction through machine learning (a case study and framework for progress)." Space Weather 19.6 (2021): e2020SW002684. [link](https://doi.org/10.1029/2020SW002684)
- Deshmukh, Varad, et al. "Decreasing False Alarm Rates in ML-based Solar Flare Prediction using SDO/HMI Data." arXiv preprint arXiv:2111.10704 (2021). [link](https://arxiv.org/pdf/2111.10704.pdf)
- Brown, Edward JE, et al. "Attention‐Based Machine Vision Models and Techniques for Solar Wind Speed Forecasting Using Solar EUV Images." Space Weather 20.3 (2022): e2021SW002976. [link](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021SW002976)
- Hu, Andong, et al. "Probabilistic prediction of Dst storms one-day-ahead using Full-Disk SoHO Images" [code](https://github.com/HuanWinter/Dst_SoHO) [link](https://arxiv.org/pdf/2203.11001.pdf)

### Other Applications

- Wright, Paul J., et al. "DeepEM: Demonstrating a Deep Learning Approach to DEM Inversion." Zenodo. [link](https://zenodo.org/record/2587015#.YMjHtZMzau8)
- Guedes dos Santos, L. F., et al. "Multi-Channel Auto-Calibration for the Atmospheric Imaging Assembly instrument with Deep Learning." AGU Fall Meeting Abstracts. Vol. 2020. 2020. [link](https://arxiv.org/abs/2012.14023)
- Jarolim, Robert, et al. "Instrument-To-Instrument translation: Instrumental advances drive restoration of solar observation series via deep learning." (2022). [link](https://assets.researchsquare.com/files/rs-1021940/v1_covered.pdf?c=1645732337) [code](https://github.com/RobertJaro/InstrumentToInstrument)

## Tools

- [Aiapy](https://gitlab.com/LMSAL_HUB/aia_hub/aiapy) - Package for analyzing calibrated (level 1) EUV imaging data from AIA.
- [IRISreader](https://github.com/i4Ds/IRISreader) - Python library that allows for efficient browsing through IRIS satellite data in order to simplify machine learning applications.
- [Global High-Resolution Hα network](http://www.bbso.njit.edu/Research/Halpha/)
- [Helioviewer](http://www.helioviewer.org) - Solar and heliospheric image visualization tool. [code](https://github.com/Helioviewer-Project/JHelioviewer-SWHV)
- [Integrated Solar Database](https://dmlab.cs.gsu.edu/solar/isd/) - Solar and heliospheric image visualization tool including image parameters with Extended Spatiotemporal Querying Capabilities.
- [integrated Space Weather Analysis (iSWA) system](https://iswa.gsfc.nasa.gov/IswaSystemWebApp/)
- [SolarMonitor](https://solarmonitor.org/) - Provides realtime information about the solar activity. The most recent images of the sun are shown, together with the description of the different NOAA AR of the day and which flares has been associated to them.
- [SpaceML](http://spaceml.org/repo) - A Machine Learning toolbox and developer community building the next generation AI applications for space science and exploration containing a set of [code examples](https://github.com/spaceml-org/helionb-sdoml).
- [Sunpy](https://sunpy.org/) - Set of tools for solar data analysis with Python.
- [SWPC CME Analysis Tool (SWPC_CAT)](https://ccmc.gsfc.nasa.gov/swpc_cat_web/) - Primary tool being used by NOAA SWPC in measuring key parameters of a Coronal Mass Ejection (CME) [code](https://github.com/nasa/ccmc-swpc-cat-web)
- [The Heliophysics KNOWledge Network](https://github.com/rmcgranaghan/Helio-KNOW)  - Collection of software and systems for improved information representation in Heliophysics.


## Videos

- [Big data 2020 Conference Talk](https://dmlab.cs.gsu.edu/bigdata/bigdata-tutorial-2020/BigData2020_Tutorial_backupTalks2.mp4) - Tutorial 6: Data Sources, Tools, and Techniques for Big Data-driven Machine Learning in Heliophysics
- [SDO 2021 Science Workshop](https://www.youtube.com/channel/UCy_NKqf3CnLGlBTBJ5DmkBA) - Recent science topics targeting SDO data
- [SpaceML Youtube Channel](https://www.youtube.com/channel/UCxI8ZDo3Gs33l3FTujx_TpQ) - machine learning toolbox and developer community building open science AI applications for space science and exploration.
