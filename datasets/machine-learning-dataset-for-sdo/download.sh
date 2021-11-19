#!/usr/bin/env bash

# -t : test mode (no download, output for verification)
# $1 : channel (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500, Bx, By, Bz)
# $2 : year (2010 - 2018)
# $3 : destination folder (/data1/data/sdo-dataset/)
# $4 : [optional] year, for range download (will download year_1 to year_2, inclusive)

test_mode=false

# Process all options supplied on the command line 
while getopts ':t' 'OPTKEY'; do
    case ${OPTKEY} in
        't')
            test_mode=true
            ;;
        '?')
            echo "INVALID OPTION -- ${OPTARG}" >&2
            exit 1
            ;;
        ':')
            echo "MISSING ARGUMENT for option -- ${OPTARG}" >&2
            exit 1
            ;;
        *)
            echo "UNIMPLEMENTED OPTION -- ${OPTKEY}" >&2
            exit 1
            ;;
    esac
done
shift $(( OPTIND - 1 ))
[[ "${1}" == "--" ]] && shift

if [ $# -lt 3 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
elif [ $# -eq 3 ]; then
  year_to=$2
elif [ $# -eq 4 ]; then
  year_to=$4
elif [ $# -gt 4 ]; then
  echo 1>&2 "$0: too many arguments"
  exit 2
fi

declare -A year_id
year_id["2010"]="vk217bh4910"
year_id["2011"]="jc488jb7715"
year_id["2012"]="dc156hp0190"
year_id["2013"]="km388vz4371"
year_id["2014"]="sr325xz9271"
year_id["2015"]="qw012qy2533"
year_id["2016"]="vf806tr8954"
year_id["2017"]="kp222tm1554"
year_id["2018"]="nk828sc2920"


instrument="AIA"
channel=$1
if [[ "$1" == B* ]]; then
  instrument="HMI"
else
  printf -v channel "%04d" $1
fi

# https://stacks.stanford.edu/file/druid:nk828sc2920/HMI_Bz_201812.tar
for y in `seq $2 ${year_to}`; do
    for i in {01..12}; do
        echo "Downloading year ${y} month ${i} from https://stacks.stanford.edu/file/druid:${year_id[${y}]}/${instrument}_${channel}_${y}${i}.tar to $3"
        if ! "$test_mode"; then
            wget -P $3 "https://stacks.stanford.edu/file/druid:${year_id[${y}]}/${instrument}_${channel}_${y}${i}.tar" --show-progress -nc
        fi
    done
done
