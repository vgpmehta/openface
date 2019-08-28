#!/bin/bash

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: compile_deps.sh"
    exit 1
fi

# boost 1.71.0 Dependency (https://github.com/zpoint/Boost-Python-Examples)
echo "Installing boost 1.71.0..."
cd boost_1_71_0
echo "using mpi ;
using gcc :  : g++ ;
using python : 3.5 : /usr/bin/python3 : /usr/include/python3.5m : /usr/local/lib ;" > ~/user-config.jam
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
sudo ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so /usr/lib/x86_64-linux-gnu/libboost_python3.so
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