#!/bin/bash

set -e

if [ -d build ]
then
    cd build
    make -j8
else
    rm -rf build
    mkdir build
    cd build
    openblas_root='/usr/local/Cellar/openblas/0.2.20_2'
    BOOST_ROOT='/usr/local/Cellar/boost@1.60/1.60.0/' cmake \
	      -D CMAKE_BUILD_TYPE=Debug \
	      -D CMAKE_PREFIX_PATH=${openblas_root} \
	      -D Qt5_DIR=/usr/local/Cellar/qt/5.10.1/lib/cmake/Qt5 ..
    make -j8
fi
