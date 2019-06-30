#!/bin/bash
#==============================================================================
# Title: install.sh
# Description: Install everything necessary for OpenFace to compile.
# Author: Daniyal Shahrokhian <daniyal@kth.se>, Tadas Baltrusaitis <tadyla@gmail.com>
# Date: 20190630
# Version : 1.03
# Usage: bash install.sh
#==============================================================================

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: install.sh"
    exit 1
fi

# Essential Dependencies
echo "Installing Essential dependencies..."

# If we're not on 18.04
if [[ `lsb_release -rs` != "18.04" ]]
  then   
    echo "Adding ppa:ubuntu-toolchain-r/test apt-repository "
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
fi

sudo apt-get -y install build-essential
sudo apt-get -y update
sudo apt-get -y install gcc-8 g++-8


sudo apt-get -y install cmake=3.10
sudo apt-get -y install zip
sudo apt-get -y install libopenblas-dev liblapack-dev
sudo apt-get -y install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev
sudo apt-get -y install libtbb2 libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
echo "Essential dependencies installed."

# OpenCV Dependency
echo "Downloading OpenCV..."
wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip 4.1.0.zip
cd opencv-4.1.0
mkdir -p build
cd build
echo "Installing OpenCV..."
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_CUDA=OFF -D BUILD_SHARED_LIBS=OFF ..
make -j4
sudo make install
cd ../..
rm 4.1.0.zip
sudo rm -r opencv-4.1.0
echo "OpenCV installed."

# dlib dependecy
echo "Downloading dlib"
wget http://dlib.net/files/dlib-19.13.tar.bz2;
tar xf dlib-19.13.tar.bz2;
cd dlib-19.13;
mkdir -p build;
cd build;
echo "Installing dlib"
cmake ..;
cmake --build . --config Release;
sudo make install;
sudo ldconfig;
cd ../..;    
rm -r dlib-19.13.tar.bz2
echo "dlib installed"

# OpenFace installation
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make
cd ..
echo "OpenFace successfully installed."