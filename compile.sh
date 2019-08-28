#!/bin/bash

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: recompile.sh"
    exit 1
fi

# boost 1.71.0 Dependency (https://github.com/zpoint/Boost-Python-Examples)
echo "Installing boost 1.71.0..."
cd boost_1_71_0
./bootstrap.sh --with-python=/usr/bin/python3 --with-python-version=3.5 --with-python-root=/usr/local/lib/python3.5 --prefix=/usr/local
sudo ./b2 install -a --with=all
sudo ldconfig
cd ..
echo "boost 1.71.0 installed."

# OpenCV Dependency
cd opencv-4.1.0
mkdir -p build
cd build
echo "Installing OpenCV..."
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_SHARED_LIBS=ON ..
make -j7
sudo make install
cd ../..
echo "OpenCV installed."

# dlib dependecy
echo "Downloading dlib"
cd dlib-19.13
rm -rf build
mkdir -p build
cd build
echo "Installing dlib"
cmake -D CMAKE_BUILD_TYPE=RELEASE -D DLIB_IN_PROJECT_BUILD=ON -D DLIB_USE_CUDA=OFF -D BUILD_SHARED_LIBS=ON ..
make -j4
sudo make install
sudo ldconfig
cd ../..  
echo "dlib installed"

# OpenFace installation
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make
cd ..
echo "OpenFace successfully installed."
