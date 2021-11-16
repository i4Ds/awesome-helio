#!/usr/bin/env bash

# -t : test mode (no extract, output for verification)
# $1 : channel (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500, Bx, By, Bz)
# $2 : year (2010 - 2018)
# $4 : destination folder (/data1/data/sdo-dataset/)
# $5 : [optional] year, for range extract (will extract year_1 to year_2, inclusive)

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

# Channel and instrument
instrument="AIA"
channel=$1
if [[ "$1" == B* ]]; then
  instrument="HMI"
else
  printf -v channel "%04d" $1
fi

if ! "$test_mode"; then
    mkdir -p $3/$1
fi

for y in `seq $2 ${year_to}`; do
    if ! "$test_mode"; then
        mkdir -p $3/$1/${y}
    fi
    for i in {01..12}; do
        if [ -f ${instrument}_${channel}_${y}${i}.tar ]; then
            echo "Extracting channel $1 year ${y} month ${i} from ${instrument}_${channel}_${y}${i}.tar to $3"
            if ! "$test_mode"; then
              tar -xzf ${instrument}_${channel}_${y}${i}.tar -C $3/$1/${y}
            fi
        fi
    done
done
