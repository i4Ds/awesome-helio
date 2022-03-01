from dotenv import load_dotenv
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
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import csv
import matplotlib.patches as mpatches
import datetime as dt
import os
import cv2
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import wkt
from shapely.geometry.point import Point
from shapely.geometry import Polygon
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import logging
from sunpy.coordinates.ephemeris import get_horizons_coord
from sunpy.net import attrs as a
from sunpy.net import Fido
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import sunpy
from sunpy.visualization.colormaps import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import numpy as np
import json
import pandas as pd
from sunpy.net import attrs as a
from sunpy.net import Fido
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, String, MetaData, DateTime, Sequence, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import insert, JSONB
import logging
import datetime

#
#   utility to generate an anomaly detection dataset from the Machine Learning Dataset for the SDO
#

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-4s %(message)s')
logger = logging.getLogger('HEKEventAnalyzer')
date_format = '%Y%m%d%H%M'


class NpEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class EVENT_TYPE:
    """
    The event types of the solar events. (More: https://www.lmsal.com/hek/VOEvent_Spec.html)
    """
    AR = 'ar'  # Active Region
    CH = 'ch'  # Coronal Hole
    FI = 'fi'  # Filament, Kanzelhöhe

    CE = 'ce'  # Coronal Mass Ejection (CME)
    FL = 'fl'  # Flare
    SG = "sg"  # Sigmoid

    @staticmethod
    def convert(et):
        return {
            'ar': EVENT_TYPE.AR,
            'ch': EVENT_TYPE.CH,
            'ce': EVENT_TYPE.CE,
            'fi': EVENT_TYPE.FI,
            'fl': EVENT_TYPE.FL,
        }.get(et, 'ar')  # default is 'ar'


class SpatioTemporalEvent:
    def __init__(self,
                 event_type: EVENT_TYPE,
                 start_time: dt.datetime,
                 end_time: dt.datetime,
                 hpc_coord: (Point, str),
                 hpc_bbox: (Polygon, str),
                 hpc_boundcc: (Polygon, str),
                 kb_archivid: str):
        """
        :param event_type: The event for which the results are returned.
        :param start_time: Start time of the event type.
        :param end_time: End Time of the event Type.
        :param hpc_coord: coordinates of the center of the bounding box.
        :param hpc_bbox: bounding box of the polygon.
        :param hpc_boundcc: polygon of the detected event (if present).
        :param kb_archivid: Unique id for each event type.
        """
        self.event_type: EVENT_TYPE = event_type
        self.start_time: dt.datetime = start_time
        self.end_time: dt.datetime = end_time
        self.hpc_coord: (Point, str) = hpc_coord
        self.hpc_bbox: (Polygon, str) = hpc_bbox
        if(hpc_boundcc):
            self.hpc_boundcc: (Polygon, str) = hpc_boundcc
        self.kb_archivid: str = kb_archivid

    @classmethod
    def from_dict(cls, instance: dict):
        obj = cls.__new__(cls)
        super(SpatioTemporalEvent, obj).__init__()
        obj.event_type = instance['event_type']
        obj.start_time = instance['event_starttime']
        obj.end_time = instance['event_endtime']
        obj.hpc_coord = instance['hpc_coord']
        obj.hpc_bbox = instance['hpc_bbox']
        if instance['hpc_boundcc'] != '':
            obj.hpc_boundcc = instance['hpc_boundcc']
        else:
            obj.hpc_boundcc = None
        obj.kb_archivid = instance['kb_archivid']
        return obj

    @property
    def event_type(self):
        return self.__event_type

    @event_type.setter
    def event_type(self, event_type):
        if isinstance(event_type, EVENT_TYPE):
            self.__event_type = event_type
        elif isinstance(event_type, str):
            self.__event_type = EVENT_TYPE.convert(event_type)

    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        if isinstance(start_time, dt.datetime):
            self.__start_time = start_time
        elif isinstance(start_time, str):
            self.__start_time = dt.datetime.strptime(
                start_time, '%Y-%m-%dT%H:%M:%S')
        else:
            raise AttributeError

    @property
    def end_time(self):
        return self.__end_time

    @end_time.setter
    def end_time(self, end_time):
        if isinstance(end_time, dt.datetime):
            self.__end_time = end_time
        elif isinstance(end_time, str):
            self.__end_time = dt.datetime.strptime(
                end_time, '%Y-%m-%dT%H:%M:%S')
        else:
            raise AttributeError

    @property
    def hpc_coord(self):
        return self.__hpc_coord

    @hpc_coord.setter
    def hpc_coord(self, hpc_coord):
        if isinstance(hpc_coord, Point):
            self.__hpc_coord = hpc_coord
        elif isinstance(hpc_coord, str):
            self.__hpc_coord = wkt.loads(hpc_coord)
        else:
            raise AttributeError

    @property
    def hpc_bbox(self):
        return self.__hpc_bbox

    @hpc_bbox.setter
    def hpc_bbox(self, hpc_bbox):
        if isinstance(hpc_bbox, Polygon):
            self.__hpc_bbox = hpc_bbox
        elif isinstance(hpc_bbox, str):
            self.__hpc_bbox = wkt.loads(hpc_bbox)
        else:
            raise AttributeError

    @property
    def hpc_boundcc(self):
        return self.__hpc_boundcc

    @hpc_boundcc.setter
    def hpc_boundcc(self, hpc_boundcc):
        if hpc_boundcc is None:
            self.__hpc_boundcc = None
        if isinstance(hpc_boundcc, Polygon):
            self.__hpc_boundcc = hpc_boundcc
        elif isinstance(hpc_boundcc, str):
            self.__hpc_boundcc = wkt.loads(hpc_boundcc)
        else:
            # raise AttributeError
            self.__hpc_boundcc = None

    @property
    def kb_archivid(self):
        return self.__kb_archivid

    @kb_archivid.setter
    def kb_archivid(self, kb_archivid):
        self.__kb_archivid = kb_archivid

    def to_dict(self):
        dict = {'event_type': self.event_type,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'hpc_coord': self.hpc_coord,
                'hpc_bbox': self.hpc_bbox,
                'hpc_boundcc': self.hpc_boundcc,
                'kb_archivid': self.kb_archivid
                }
        return dict


def convert_boundingpoints_to_pixelunit(polygon: Polygon, cdelt1, cdelt2, crpix1, crpix2, original_w, shrinkage_ratio=1):
    """
    This method converts the points coordinates from arc-sec unit to pixel unit, and meanwhile
    makes 2 modifications:
        1. Vertical mirror of the points (this is required if JPG format is being used)
        2. Shrinkage of points (required if downsized images are being used.)

    :param polygon: a list of points forming a closed shape.
    :param cdelt1: fits/jp2 header information to scale in x direction
    :param cdelt2: fits/jp2 header information to scale in y direction
    :param crpix1: fits/jp2 header information to shift in x direction
    :param crpix2: fits/jp2 header information to shift in y direction
    :param original_w: the width of the original images. It is assumed that the images
    are in square shape, hence width and height are equal.
    :param shrinkage_ratio: a float point that indicates the ratio (original_w/new_size).
    For example, for 512 X 512 image, it should be 8.0.
    :return: a polygon object (from Shapely package) and None if the list was empty. If you need
            a list of tuples (x, y) instead, you can convert it using `poly = poly.exterior.coords'
    """
    points = polygon.exterior.coords
    b = [(float(v[0]) / cdelt1 + crpix1, float(v[1]) / cdelt2 + crpix2)
         for v in points]

    # TODO for the currated image params dataset these need to be mirror horizontally, why not here?
    # TODO is even a shift required here, as the images have been shifted and rescaled already...at any rate the vents would need to be shifted in the same way
    # b = [(v[0] / shrinkage_ratio, (original_w - v[1]) / shrinkage_ratio)
    b = [(v[0] / shrinkage_ratio, (v[1]) / shrinkage_ratio)
         for v in b]

    return Polygon(b)


def convert_events_to_pixelunits(events_df: pd.DataFrame, img_header, original_width=4096, shrinkage_factor=8):
    all_polygons = []
    all_bboxes = []

    for i, event in events_df.iterrows():
        ste = SpatioTemporalEvent.from_dict(event)

        if(ste.hpc_boundcc is not None):
            # hpc_boundcc is optional
            poly_converted = convert_boundingpoints_to_pixelunit(polygon=ste.hpc_boundcc,
                                                                 cdelt1=img_header['CDELT'],
                                                                 cdelt2=img_header['CDELT'],
                                                                 crpix1=img_header['X0'],
                                                                 crpix2=img_header['Y0'],
                                                                 original_w=original_width,
                                                                 shrinkage_ratio=shrinkage_factor)
            all_polygons.append(poly_converted)

        bbox_converted = convert_boundingpoints_to_pixelunit(polygon=ste.hpc_bbox,
                                                             cdelt1=img_header['CDELT'],
                                                             cdelt2=img_header['CDELT'],
                                                             crpix1=img_header['X0'],
                                                             crpix2=img_header['Y0'],
                                                             original_w=original_width,
                                                             shrinkage_ratio=shrinkage_factor)

        all_bboxes.append(bbox_converted)

    return (all_polygons, all_bboxes)


def get_meta_info(image_path, timestamp):
    # https://docs.sunpy.org/en/stable/api/sunpy.coordinates.ephemeris.get_horizons_coord.html#sunpy.coordinates.ephemeris.get_horizons_coord

    coords = get_horizons_coord('SDO', index.timestamp)
    # TODO how to obtain just the FITS header information?
    # QUALITY,DSUN,X0,R_SUN,Y0,CDELT,FILE_NAME
    # 1073741824,147126420000.0,2053.23,1628.7837,2045.8,0.599076,2012-01-02T060000__171.jpeg
    header = {
        'DSUN': 147126420000.0,
        'X0': 2053.23,
        'Y0': 2045.8,
        "R_SUN": 1628.7837,
        "CDELT": 0.599076,
        "timestamp": index.timestamp
    }

    return header


def get_date_ranges(start, end, freq="d"):
    # retrieve daily ranges between start and end to not overload the HEK API

    dates = pd.date_range(start=start, end=end,
                          freq=freq).to_pydatetime().tolist()

    dates[0] = start
    if(dates[len(dates) - 1] < end):
        dates.append(end)

    ranges = []
    for i in range(0, len(dates) - 1):
        t_start = dates[i]
        if(i < len(dates) - 1):
            t_end = dates[i + 1]
        else:
            t_end = end

        ranges.append((t_start, t_end))

    return ranges


def load_events_from_hek(start: dt.datetime, end: dt.datetime, event_type: str):
    """
    Retrieves a set of events from HEK and stores it in the local database
    """
    logger.info(
        f"starting to load events from HEK between {start} and {end} for type {event_type} from HEK")
    date_ranges = get_date_ranges(start, end)
    total_events = 0
    all_events_dfs = []
    for t_start, t_end in date_ranges:
        logger.debug(
            f"loading events from HEK between {t_start} and {t_end} for type {event_type} from HEK")
        event_query = a.hek.EventType(event_type)
        result = Fido.search(a.Time(t_start, t_end), event_query)
        col_names = [name for name in result["hek"].colnames if len(
            result["hek"][name].shape) <= 1]
        events_df = result["hek"][col_names].to_pandas()
        total_events = total_events + len(events_df)
        all_events_dfs.append(events_df)

    logger.info(f"retrieved a total of {total_events} from HEK")
    return pd.concat(all_events_dfs)


def load_boxes(src_img_full_path, index_path, hek_event_type='AR', instrument="AIA", allowed_time_diff_seconds=300):
    meta = get_meta_info(src_img_full_path, index_path)
    timestamp = meta["timestamp"]
    start = timestamp - dt.timedelta(seconds=allowed_time_diff_seconds)
    end = timestamp + dt.timedelta(seconds=allowed_time_diff_seconds)
    events_df = load_events_from_hek(start, end, hek_event_type)

    if len(events_df) < 1:
        logger.warning(f"no events found")
        return None, None, None

    # filter events that were observed in the respective wavelength, possibly also filter by feature extraction method
    events_df = events_df[events_df['obs_observatory'].str.contains("SDO")]
    logger.info(f"after filter {len(events_df)} events")

    hek_bboxes, hek_polygons = convert_events_to_pixelunits(events_df, meta)
    return events_df, hek_bboxes, hek_polygons


# Channels that correspond to HMI Magnetograms
HMI_WL = ['bx', 'by', 'bz']
# A colormap for visualizing HMI
HMI_CM = LinearSegmentedColormap.from_list(
    "bwrblack", ["#0000ff", "#000000", "#ff0000"])


def channelNameToMap(name):
    """Given channel name, return colormap"""
    return HMI_CM if name in HMI_WL else cm.cmlist.get('sdoaia%d' % int(name))


def getClip(X, name):
    """Given an image and the channel name, get the right clip"""
    return getSignedPctClip(X) if name in HMI_WL else getPctClip(X)


def getPctClip(X):
    """Return the 99.99th percentile"""
    return (0, np.quantile(X.ravel(), 0.999))


def getSignedPctClip(X):
    """Return the 99.99th percentile by magnitude, but symmetrize it so 0 is in the middle"""
    v = np.quantile(np.abs(X.ravel()), 0.999)
    return (-v, v)


def vis(X, cm, clip):
    """Given image, colormap, and a clipping, visualize results"""
    Xc = np.clip((X-clip[0])/(clip[1]-clip[0]), 0, 1)
    Xcv = cm(Xc)
    return (Xcv[:, :, :3]*255).astype(np.uint8)


def display_img(img_path, hek_bboxes, hek_polygons, channelName):
    if hek_bboxes is None or hek_polygons is None:
        return
    plt.clf()
    X = np.load(img_path)['x'].astype(np.float64)
    V = vis(X, channelNameToMap(channelName), getClip(X, channelName))

    img = Image.fromarray(V)
    # img = img.convert('RGB')
    img_draw = ImageDraw.Draw(img)

    fig = plt.figure(figsize=(12, 12))

    for poly in hek_polygons:
        poly = poly.exterior.coords
        img_draw.line(poly, fill="red", width=1)
        for point in poly:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="red")

    for bbox in hek_bboxes:
        bbox = bbox.exterior.coords
        img_draw.line(bbox, fill="blue", width=1)
        for point in bbox:
            img_draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill="blue")

    plt.axis('off')
    title = img_path.name
    plt.title(title)

    red_patch = mpatches.Patch(color='red', label='HEK bounding box')
    blue_patch = mpatches.Patch(color='blue', label='HEK exact bounding box')

    plt.legend(handles=[red_patch, blue_patch])

    plt.imshow(img)


def save_mask_file(img_path, target_path, hek_polygons, channelName):
    if hek_polygons is None:
        return
    plt.clf()
    X = np.load(img_path)['x'].astype(np.float64)
    V = vis(X, channelNameToMap(channelName), getClip(X, channelName))

    img = Image.fromarray(V)
    img_draw = ImageDraw.Draw(img)

    fig = plt.figure(figsize=(12, 12))

    for poly in hek_polygons:
        poly = poly.exterior.coords
        img_draw.polygon(poly, fill="red")

    plt.axis('off')
    title = img_path.name
    plt.title(title)
    fig.savefig(target_path)


def store_events(connection_string: str, events_df):
    db = create_engine(connection_string)
    meta = MetaData(db)
    events_table = Table('hek_events', meta,
                         Column('event_id', Integer, Sequence(
                             'event_id_seq'), primary_key=True),
                         Column('event_type', String, nullable=False),
                         Column('event_starttime',
                                DateTime, nullable=False),
                         Column('event_endtime',
                                DateTime, nullable=False),
                         Column('obs_observatory', String),
                         Column('obs_instrument', String),
                         Column('obs_channelid', String),
                         Column('kb_archivid', String,
                                nullable=False),
                         Column('hpc_bbox', String),
                         Column('hpc_boundcc', String),
                         Column('hpc_coord', String),
                         Column('full_event', JSONB),
                         UniqueConstraint('kb_archivid', name='uix_kb_archivid'))
    events_table.create(checkfirst=True)

    with db.connect() as conn:
        for idx, event in events_df.iterrows():
            full_event = events_df.iloc[idx].to_dict()
            full_event_json = json.dumps(full_event, cls=NpEncoder)
            insert_statement = insert(events_table).values(
                event_type=event["event_type"],
                event_starttime=event["event_starttime"],
                event_endtime=event["event_endtime"],
                obs_observatory=event["obs_observatory"],
                obs_instrument=event["obs_instrument"],
                obs_channelid=event["obs_channelid"],
                kb_archivid=event["kb_archivid"],
                hpc_bbox=event["hpc_bbox"],
                hpc_boundcc=event["hpc_boundcc"],
                hpc_coord=event["hpc_coord"],
                # https://www.compose.com/articles/using-json-extensions-in-postgresql-from-python-2/
                # https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/
                full_event=full_event_json)

            update_dict = {
                c.name: c for c in insert_statement.excluded if not c.primary_key}
            insert_statement = insert_statement.on_conflict_do_update(
                index_elements=['kb_archivid'],
                set_=update_dict)
            conn.execute(insert_statement)


def find_events_at(timestamp, event_types=None, observatory=None, instrument=None, allowed_time_diff_seconds=30) -> pd.DataFrame:
    with db.connect() as conn:
        select_statement = events_table.select()
        select_statement = select_statement.where(
            events_table.c.event_starttime <= timestamp + datetime.timedelta(seconds=allowed_time_diff_seconds))
        select_statement = select_statement.where(
            events_table.c.event_endtime >= timestamp - datetime.timedelta(seconds=allowed_time_diff_seconds))

        if instrument is not None:
            select_statement = select_statement.where(
                events_table.c.obs_instrument == instrument)

        if observatory is not None:
            select_statement = select_statement.where(
                events_table.c.obs_observatory == observatory)

        if event_types is not None:
            select_statement = select_statement.where(
                events_table.c.event_type.in_(event_types))

        result_set = conn.execute(select_statement)
        df = pd.DataFrame(result_set)
        if result_set.rowcount > 0:
            df.columns = result_set.keys()
        logger.info(f"retrieved {len(df)} events from local database")

        return df


def fetch_events(data_path, target_path, connection_string, event_types=["AR", "CH"]):
    index_df = load_index(data_path, target_path)
    first_known_date = index_df.index.min()
    last_known_date = index_df.index.max()

    for event_type in event_types:
        events_df = load_events_from_hek(
            first_known_date, last_known_date, event_type)
        store_events(connection_string, events_df)


def load_index(data_path, target_path):
    index_path = data_path / Path("index.csv")
    index_df = pd.read_csv(index_path)
    index_df["timestamp"] = pd.to_datetime(
        index_df['timestamp'], format=ISO_FORMAT)
    index_df = index_df.set_index('timestamp', drop=False)
    return index_df


def create_ae_dataset(data_path, target_path):
    index_df = load_index(data_path, target_path)
    for idx, row in index_df.iterrows():

        meta = get_meta_info(row["path"], index_path)
        events_df = find_events_at()
        if len(events_df) < 1:
            logger.warning(f"no events found")
            continue

        # filter events that were observed in the respective wavelength, possibly also filter by feature extraction method
        events_df = events_df[events_df['obs_observatory'].str.contains("SDO")]
        logger.info(f"after filter {len(events_df)} events")

        hek_bboxes, hek_polygons = convert_events_to_pixelunits(
            events_df, meta)
        save_mask_file(src_img_full_path, hek_bboxes, channelName=171)


load_dotenv()


@click.command()
@click.option('--data-dir',  type=click.Path(exists=True, file_okay=False, readable=True), default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo_extracted", help='directory containing the compressed dataset (tar files)')
@click.option('--target-dir',  type=click.Path(writable=True), default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo_extracted_ae", help='target path for the extracted dataset')
@click.option("--db-connection-string", default=lambda: os.environ.get("DB_CONNECTION_STRING", ""))
def cli(data_dir, target_dir, db_connection_string, should_fetch_events=False):
    file_names = set(Path(data_dir).rglob(f'*.tar'))

    print(f"starting to create AE dataset in {target_dir}")
    if should_fetch_events:
        fetch_events(data_dir, target_dir, db_connection_string)
    create_ae_dataset(data_dir, target_dir, db_connection_string)


if __name__ == "__main__":
    cli()
