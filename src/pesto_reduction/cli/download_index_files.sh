#!/bin/bash

BASE_URL="http://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE"

for i in $(seq -w 0 47); do
    FILE="index-5200-${i}.fits"
    echo "Downloading $FILE ..."
    wget -c "${BASE_URL}/${FILE}"
done