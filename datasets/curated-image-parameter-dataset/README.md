# Curated Image Parameter Dataset <!-- omit in toc -->

Includes an image parameter dataset extracted from the Solar Dynamics Observatory (SDO) mission’s AIA instrument computed for the period of 2011-01 through now, with the cadence of 6 minutes, for 9 wavelength channels. 

>We provide a massive image parameter dataset extracted from the Solar Dynamics Observatory (SDO) mission’s AIA instrument, for the period of January 2011 through the current date, with the cadence of six minutes, for nine wavelength channels. Towards better results in the region classification of solar events, in this work, we improve upon the performance of a set of ten image parameters when utilized in the task of classifying regions of the AIA images. This is accomplished through an in depth analysis of various assumptions made in the calculation of these parameters, in order to find areas of improving the outcome of the stated classification task. Then, where possible, a method for finding an appropriate settings for the parameter calculations was devised, as well as a validation task that is used to show our improved results. This process is repeated for each of the nine different wavelength channels that are included in our analysis. In addition to investigating the effects of assumptions made during the calculation process, we also include comparisons of JP2 and FITS image formats in a pixel-based, supervised classification task, by tuning the parameters specific to the format of the images from which they are extracted. The results of these comparisons show that utilizing JP2 images, which are significantly smaller files, is not detrimental to the region classification task that these parameters were originally intended for. Finally, we compute the tuned parameters on the AIA images and to make the resultant dataset easily accessible for others, we provide this public API for random access to the calculated parameters.

1 TiB of data per year

- [Data Access](#data-access)
  - [API](#api)
  - [sdo-cli wrapper](#sdo-cli-wrapper)
  - [NAS](#nas)
- [Cite](#cite)

## Data Access

### API

The [GSU DMLAB](http://dmlab.cs.gsu.edu/dmlabapi/isd_docs.html) provides an API for retrieving data from the Curated Image Parameter dataset.

### sdo-cli wrapper 

[sdo-cli](https://github.com/i4Ds/sdo-cli) is a wrapper for working with the DMLAB API and can be used as follows:

```
sdo-cli data download --path='./data/aia_2012_01_171_193' --start='2012-01-01T00:00:00' --end='2012-01-31T23:59:59' --freq='6h' --wavelength='171' --wavelength='193' --metadata
```

### NAS

A subset of the data can be found on the FHNW NAS under data02/sdo/curated-image-parameter-dataset.

## Cite

```
@article{ahmadzadeh2019curated,
  title={A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission},
  author={Ahmadzadeh, Azim and Kempton, Dustin J and Angryk, Rafal A},
  journal={The Astrophysical Journal Supplement Series},
  volume={243},
  number={1},
  pages={18},
  year={2019},
  publisher={IOP Publishing}
}
```