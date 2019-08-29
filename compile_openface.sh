#!/bin/bash

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: compile_openface.sh"
    exit 1
fi

# OpenFace installation
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make
# copying models in FeatureExtractionPython folder
cp -r bin/model/ exe/FeatureExtractionPython/model/
mkdir -p exe/FeatureExtractionPython/classifiers/
cp ../lib/3rdParty/OpenCV/classifiers/haarcascade_frontalface_alt.xml exe/FeatureExtractionPython/classifiers/
echo "OpenFace successfully installed."
