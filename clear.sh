#!/bin/bash

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: recompile.sh"
    exit 1
fi

rm -rf boost_1_71_0.tar.bz2*
rm -rf 4.1.0.zip*
rm -rf dlib-19.13.tar.bz2*

sudo rm -rf cmake_tmp
sudo rm -rf boost_1_71_0
sudo rm -rf dlib-19.13
sudo rm -rf opencv-4.1.0
sudo rm -rf build
