# Awesome Helio  <!-- omit in toc -->

A curated list of datasets, tools and papers for machine learning in heliophysics.

## About

The primary goal of `awesome-helio` is to provide a compiled list for all the great resources contributing to the state of the art in heliophysics that are out there but are often hard to find. `awesome-helio` is mainly focussed on machine learning groups but can also provide important insights for other users.

## Contributing

Please take a quick look at the [contribution guidelines](https://github.com/i4Ds/awesome-helio/blob/master/CONTRIBUTING.md) first.

**_If you find any datasets, tools and papers that is missing or is not a good fit, please submit a pull request to improve this file. Thank you!_**

### Contents  <!-- omit in toc -->

- [About](#about)
- [Contributing](#contributing)
- [Missions](#missions)
- [Datasets](#datasets)
  - [SDO](#sdo)
  - [IRIS](#iris)
- [Tools](#tools)
- [Papers](#papers)
    - [Datasets](#datasets-1)

## Missions

There is a fleet of spacecrafts strategically placed throughout our heliosphere:

<span class="img_container center" style="display: block;">
    <img alt="Heliophysics Mission Fleet" src="../main/images/helio-fleet-01-2021.jpg?raw=true" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <span class="img_caption" style="display: block; text-align: center;">Heliophysics Mission Fleet, Source <a href="https://science.nasa.gov/heliophysics/mission-fleet-diagram">NASA</a></span>
</span>
</br>

A comprehensive list can be found [here](https://science.nasa.gov/missions-page?field_division_tid=5&field_phase_tid=All).

## Datasets

### SDO 

- [Curated Image Parameter Dataset](http://dmlab.cs.gsu.edu/dmlabapi/isd_docs.html) - Massive image parameter dataset extracted from the Solar Dynamics Observatory (SDO) mission’s AIA instrument, for the period of January 2011 through the current date, with the cadence of six minutes, for nine wavelength channels.
- [JSOC](http://jsoc.stanford.edu/) - Data products from the Solar Dynamics Observatory, as well as certain other missions and instruments, are available from the JSOC database.
- [Machine Learning Data Set for NASA's Solar Dynamics Observatory](https://purl.stanford.edu/nk828sc2920) - A curated dataset from the NASA Solar Dynamics Observatory (SDO) mission in a format suitable for machine learning research.
- [SDOBenchmark](https://i4ds.github.io/SDOBenchmark/) - Machine learning image dataset for the prediction of solar flares.

### IRIS

TODO


## Tools

- [Aiapy](https://gitlab.com/LMSAL_HUB/aia_hub/aiapy) - Package for analyzing calibrated (level 1) EUV imaging data from AIA.
- [IRISreader](https://github.com/i4Ds/IRISreader) - Python library that allows for efficient browsing through IRIS satellite data in order to simplify machine learning applications.
- [Helioviewer](http://www.helioviewer.org) - Solar and heliospheric image visualization tool.
- [Sunpy](https://sunpy.org/) - Set of tools for solar data analysis with Python.

## Papers

1. Kucuk, Ahmet, Juan M. Banda, and Rafal A. Angryk. "Solar event classification using deep convolutional neural networks." International Conference on Artificial Intelligence and Soft Computing. Springer, Cham, 2017. [link](https://www.researchgate.net/publication/317570870)
2. Illarionov, Egor A., and Andrey G. Tlatov. "Segmentation of coronal holes in solar disc images with a convolutional neural network." Monthly Notices of the Royal Astronomical Society 481.4 (2018): 5014-5021. [link](https://arxiv.org/abs/1809.05748)
3. Armstrong, John A., and Lyndsay Fletcher. "Fast solar image classification using deep learning and its importance for automation in solar physics." Solar Physics 294.6 (2019): 1-23. [link](https://link.springer.com/article/10.1007/s11207-019-1473-z)
4. Love, Teri, Thomas Neukirch, and Clare E. Parnell. "Analyzing AIA Flare Observations Using Convolutional Neural Networks." Frontiers in Astronomy and Space Sciences 7 (2020): 34. [link](https://doi.org/10.3389/fspas.2020.00034)
5. McGregor, Sean, et al. "Flarenet: A deep learning framework for solar phenomena prediction." Neural Information Processing Systems (NIPS) 2017 workshop on Deep Learning for Physical Sciences (DLPS), Long Beach, CA, US. 2017. [link](http://solardynamo.org/publications/McGregor_etal_NIPS_2017.pdf)
6. Schuh, Michael A., Dustin Kempton, and Rafal A. Angryk. "A Region-Based Retrieval System for Heliophysics Imagery." FLAIRS Conference. 2017. [link](https://www.researchgate.net/profile/Dustin-Kempton-2/publication/317385457_A_Region-based_Retrieval_System_for_Heliophysics_Imagery/links/5cf7a11692851c4dd02a3da9/A-Region-based-Retrieval-System-for-Heliophysics-Imagery.pdf)
7. Nagem, Tarek AM, et al. "Deep learning technology for predicting solar flares from (Geostationary Operational Environmental Satellite) data." (2018) [link](https://www.researchgate.net/publication/322924477_Deep_Learning_Technology_for_Predicting_Solar_Flares_from_Geostationary_Operational_Environmental_Satellite_Data?enrichId=rgreq-c8121ef3caa7c31906fde5bb9c53e014-XXX&enrichSource=Y292ZXJQYWdlOzMyMjkyNDQ3NztBUzo1OTAyMDE3ODA5NzM1NjhAMTUxNzcyNjQ3ODAyNA==&el=1_x_2&_esc=publicationCoverPdf)
8. Kempton, Dustin J., Michael A. Schuh, and Rafal A. Angryk. "Tracking solar phenomena from the sdo." The Astrophysical Journal 869.1 (2018): 54. [link](https://iopscience.iop.org/article/10.3847/1538-4357/aae9e9)


#### Datasets

1. Galvez, Richard, et al. "A machine-learning data set prepared from the NASA solar dynamics observatory mission." The Astrophysical Journal Supplement Series 242.1 (2019): 7. [link](https://arxiv.org/abs/1903.04538)
2. Ahmadzadeh, Azim, Dustin J. Kempton, and Rafal A. Angryk. "A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission." The Astrophysical Journal Supplement Series 243.1 (2019): 18. [link](https://arxiv.org/abs/1906.01062)

