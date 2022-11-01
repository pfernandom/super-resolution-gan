#!/bin/bash
# Takes a 1080p video and generates 3 folders of images to be used for training

set -e

if [ "$#" != 2 ]; then
  echo "Usage: $0 input_video_1080p.mp4"
  exit 1
fi

INPUT_VIDEO=$1
FRAMES=$2

BASE=`basename $INPUT_VIDEO`
VIDEO_EXT="${BASE##*.}"
OUTPUT_DIR="${BASE%.*}"
REF_DIR=${OUTPUT_DIR}/

FULL_RES="1920:1409"
LOW_RES="654:480"

mkdir -p ${REF_DIR} ${LOW_RES_DIR} ${SCALED_DIR}

# Hide excessive output, and overwrite existing files
DEFAULT_FLAGS="-hide_banner -loglevel warning -y"

echo "Splitting video into images..."
echo ${FRAMES}

# Split input_video into pngs, labelled 0001.png, 0002.png etc, where -r 1 specifies 1 frame per second
ffmpeg ${DEFAULT_FLAGS} -i ${INPUT_VIDEO} -r ${FRAMES} ${REF_DIR}/%04d.png

