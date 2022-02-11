from pathlib import Path
import os
import re
import tarfile
import csv
import datetime as dt
import argparse
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
import pandas as pd
import click
import pathlib

#
#    small data parsing utility for the Machine Learning Dataset for SDO
#

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def extract(file_names, target_dir):
    for idx, file_name in enumerate(file_names):
        print(f"processing {file_name}, {idx+1} of {len(file_names)}")
        parts = file_name.name.split("_")
        instrument = parts[0]
        channel = parts[1]
        year = parts[2][0:4]
        month = parts[2][4:6]

        try:
            with tarfile.open(file_name) as tar_file:
                tar_file.extractall(target_dir / Path(year))
        except Exception as e:
            print(e)


def index(data_dir):
    data_files = list(Path(data_dir).rglob(f'*.npz'))
    csv_fieldnames = ['path', 'file_name',
                      "instrument", "channel",  "timestamp"]

    print(f"indexing {data_dir}, with {len(data_files)} files")
    index_path = Path(data_dir) / "index.csv"
    with open(index_path, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=csv_fieldnames)
        writer.writeheader()

        # filenames are formatted like: AIA20100615_0800_4500.npz
        date_format = '%Y%m%d%H%M'
        for data_file in data_files:
            file_name_parts = data_file.name.split("_")
            instrument = file_name_parts[0][0:3]
            channel = file_name_parts[2].split(".")[0]
            datetime_str = file_name_parts[0][3:11] + file_name_parts[1][0:4]
            timestamp = dt.datetime.strptime(datetime_str, date_format)

            label = {}
            label["path"] = data_file
            label["file_name"] = data_file.name
            label["instrument"] = instrument
            label["channel"] = channel
            label["timestamp"] = timestamp.isoformat()

            writer.writerow(label)


def fetch_metadata(target_path):
    index_path = target_path / Path("index.csv")
    index_df = pd.read_csv(index_path)
    index_df["timestamp"] = pd.to_datetime(
        index_df['timestamp'], format=ISO_FORMAT)
    index_df = index_df.set_index('timestamp', drop=False)
    first_known_date = index_df.index.min()
    last_known_date = index_df.index.max()
    # pre-cache goes metadata
    fetch_goes_metadata(first_known_date, last_known_date, target_path)
    for idx, row in index_df.iterrows():
        goes = get_goes_at(row["timestamp"], target_path)
        # On each GOES satellite there are two X-ray Sensors (XRS) which provide solar X-ray irradiances for the wavelength bands of
        # 0.5 to 4A (short channel) and 1 to 8A (long channel)
        if goes is not None:
            index_df.loc[idx, "xrsa"] = goes["xrsa"]
            index_df.loc[idx, "xrsb"] = goes["xrsb"]

    index_df.to_csv(target_path / Path("index_goes.csv"), index=True,
                    date_format=ISO_FORMAT)


def get_date_ranges(start, end):
    # retrieve monthly ranges between start and end to not overload the HEK API
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    adjusted_start = dt.datetime(start.year, start.month, 1, 0, 0, 0)
    adjusted_end = dt.datetime(end.year, end.month, end.day, 23, 59, 59)

    start_dates = pd.date_range(start=adjusted_start, end=adjusted_end,
                                freq="MS").to_pydatetime().tolist()

    end_dates = pd.date_range(start=adjusted_start, end=adjusted_end,
                              freq="M").to_pydatetime().tolist()

    if len(start_dates) == 0:
        return [(adjusted_start, adjusted_end)]

    ranges = []
    for s, e in zip(start_dates, end_dates):
        ranges.append((s, e))

    if len(ranges) == 0:
        return [(adjusted_start, adjusted_end)]

    return ranges


goes_cache = {}


def fetch_goes_metadata(start, end, path):
    # NOTE it can happen that sunpy will download invalid files during rate limiting, find them using grep -r "Too many" ~/sunpy/data/*
    # https://docs.sunpy.org/en/v3.0.0/generated/gallery/acquiring_data/goes_xrs_example.html
    # https://www.ngdc.noaa.gov/stp/satellite/goes-r.html
    for t_start, t_end in get_date_ranges(start, end):
        goes_dir = Path(path) / Path("goes_cache") / Path(str(t_start.year)) /\
            Path(str(t_start.month))
        goes_path = goes_dir / Path('goes_ts.csv')
        if not pathlib.Path.exists(goes_path):
            try:
                print(f"retrieving goes data for {goes_path}")
                os.makedirs(goes_dir, exist_ok=True)
                satellite = get_goes_satellite(t_start)
                search_result = Fido.search(
                    a.Time(t_start, t_end), a.Instrument.xrs, satellite)
                download_result = Fido.fetch(search_result)
                goes_ts = ts.TimeSeries(download_result)
                if isinstance(goes_ts, list) and len(goes_ts) > 0:
                    frames = []
                    for goes_ts_frm in goes_ts:
                        frames.append(goes_ts_frm.to_dataframe())
                    goes_ts_df = pd.concat(frames)
                else:
                    goes_ts_df = goes_ts.to_dataframe()

                goes_ts_df.index.name = 'timestamp'
                goes_ts_df.to_csv(goes_path, index=True,
                                  date_format=ISO_FORMAT)
            except Exception as e:
                print(e)
                goes_ts_df = pd.DataFrame()
        else:
            try:
                print(f"goes data for {goes_path} already present")
                if goes_path in goes_cache:
                    goes_ts_df = goes_cache[goes_path]
                else:
                    goes_ts_df = pd.read_csv(goes_path)
                    goes_ts_df["timestamp"] = pd.to_datetime(
                        goes_ts_df['timestamp'], format=ISO_FORMAT)
                    goes_ts_df = goes_ts_df.set_index('timestamp', drop=False)
                    goes_ts_df = goes_ts_df.sort_index()
                    goes_cache[goes_path] = goes_ts_df

            except Exception as e:
                print(e)
                goes_ts_df = pd.DataFrame()

    return goes_ts_df


def get_goes_satellite(date):
    # https://www.ngdc.noaa.gov/stp/satellite/goes-r.html
    # https://www.ngdc.noaa.gov/stp/satellite/goes/index.html

    # Since March 2020, data prior to GOES 15 (incl) is no longer supported by NOAA and GOES 16 and 17 data is now provided.
    # GOES 16 and 17 are part of the GOES-R series and provide XRS data at a better time resolution (1s).
    # GOES 16 has been taking observations from 2017, and GOES 17 since 2018

    if date >= dt.datetime(2018, 6, 1):
        return a.goes.SatelliteNumber(17)

    if date >= dt.datetime(2017, 2, 7):
        return a.goes.SatelliteNumber(16)

    if date <= dt.datetime(2010, 9, 1):
        return a.goes.SatelliteNumber(14)

    return a.goes.SatelliteNumber(15)


def get_goes_at(at, goes_dir):
    # https://github.com/sunpy/sunpy/blob/master/sunpy/timeseries/sources/goes.py
    # https://ngdc.noaa.gov/stp/satellite/goes/doc/GOES_XRS_readme.pdf
    goes_ts_df = fetch_goes_metadata(at, at, goes_dir)
    goes_at_time = goes_ts_df.iloc[goes_ts_df.index.get_loc(
        at, method='nearest')]

    # TODO check if flux values need to be rescaled
    # NOAA have recently re-processed the GOES 13, 14 and 15 XRS science quality data,
    # such that the SWPC scaling factor has been removed.
    # This means that the fluxes will have a different values, and so will flare peak fluxes from previous 13, 14 and 15 XRS data

    # TODO To get the true fluxes, divide the short band flux by 0.85 and divide the long band flux by 0.7.
    ts = goes_at_time["timestamp"]
    diff = (ts - at).total_seconds()
    # if diff is bigger than 60, ignore..
    if diff >= 60:
        print(f"goes diff too large, goes at {ts} wanted {at}, diff {diff}")
        return None

    return goes_at_time


@click.command()
@click.option('--should-extract', default=False, is_flag=True, help='Value to indicate weather extraction should run.')
@click.option('--data-dir',  type=click.Path(exists=True, file_okay=False, readable=True), default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo", help='directory containing the compressed dataset (tar files)')
@click.option('--target-dir',  type=click.Path(writable=True), default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo_extracted", help='target path for the extracted dataset')
def cli(should_extract, data_dir, target_dir):
    file_names = set(Path(data_dir).rglob(f'*.tar'))

    if should_extract:
        print(f"extracting files from {data_dir} to {target_dir}")
        extract(file_names, target_dir)

    print(f"indexing files in {target_dir}")
    index(target_dir)
    fetch_metadata(target_dir)


if __name__ == "__main__":
    cli()
