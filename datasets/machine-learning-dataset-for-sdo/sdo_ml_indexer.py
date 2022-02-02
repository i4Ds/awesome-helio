from pathlib import Path
import os
import tarfile
import csv
import datetime as dt
import argparse

#
#    small data parsing utility for the Machine Learning Dataset for SDO
#


def extract(file_names, target_dir):
    for idx, file_name in enumerate(file_names):

        print(f"processing {file_name}, {idx+1} of {len(file_names)}")
        parts = file_name.name.split("_")
        instrument = parts[0]
        channel = parts[1]
        year = parts[2][0:4]
        month = parts[2][4:6]

        with tarfile.open(file_name) as tar_file:
            tar_file.extractall(target_dir / Path(year))


def index(data_dir):
    data_files = Path(data_dir).rglob(f'*.npz')
    csv_fieldnames = ['path', 'file_name',
                      "instrument", "channel",  "timestamp"]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', default='True',
                        help='bool value to indicate weather extraction should run')
    parser.add_argument('--data_dir', default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo",
                        help='directory containing the compressed dataset (tar files)')
    parser.add_argument(
        '--target_dir', default="/mnt/data02/sdo/stanford_machine_learning_dataset_for_sdo_extracted", help='target path for the extracted dataset')
    parser.add_argument
    args = parser.parse_args()

    data_dir_str = args.data_dir
    target_dir_str = args.target_dir

    data_dir_str = os.path.expanduser(data_dir_str)
    target_dir_str = os.path.expanduser(target_dir_str)
    if os.path.exists(data_dir_str) and not os.path.isdir(data_dir_str):
        raise ValueError(data_dir_str + " is not a directory")

    data_dir = Path(data_dir_str)
    target_dir = Path(target_dir_str)
    file_names = set(Path(data_dir).rglob(f'*.tar'))

    if args.extract:
        print(f"extracting files from {data_dir} to {target_dir}")
        extract(file_names, target_dir)

    print(f"indexing files in {target_dir}")
    index(target_dir)
