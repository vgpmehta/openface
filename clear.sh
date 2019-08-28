#!/bin/bash

# Exit script if any command fails
set -e 
set -o pipefail

if [ $# -ne 0 ]
  then
    echo "Usage: recompile.sh"
    exit 1
fi

rm -r boost_1_71_0.tar.bz2*
rm -r 4.1.0.zip*
rm -r dlib-19.13.tar.bz2*

rm -r cmake_tmp
rm -r boost_1_71_0
rm -r dlib-19.13
rm -r opencv-4.1.0