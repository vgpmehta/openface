cmake_minimum_required(VERSION 3.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/modules/")

project(PyOpenFace VERSION 0.0.1)

# NOTE: This is separate from the main CMake build script so that there's no dependency on
# Python to build, and so that building/installing is done using the normal method of pip or
# python setup.py install (makes building binary wheels for multiple Python versions easier)


# Find libraries the static OpenFace libraries need to work properly

find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect)
find_package(OpenBLAS REQUIRED)
find_package(Boost 1.5.9 COMPONENTS filesystem system)
find_package(dlib 19.13)
if(${dlib_FOUND})
    if (NOT TARGET dlib)
        add_library(dlib INTERFACE IMPORTED GLOBAL)
    endif()
endif()


# Find the pre-compiled OpenFace libraries (expecting OpenFace already built)

find_library(
    LANDMARK_LIB
    NAME LandmarkDetector
    HINTS ../build/lib/local/LandmarkDetector/
    NO_DEFAULT_PATH
)


# Add required include directories

include_directories(
    ../lib/3rdParty/dlib/include/
    ../lib/3rdParty/OpenCV/include/
    ../lib/local/LandmarkDetector/include/
)


# Actually create the pybind module for OpenFace

add_subdirectory(pybind11)
pybind11_add_module(
    _openface MODULE src/openface.cpp
    ../lib/local/LandmarkDetector/include/LandmarkDetectorModel.h
    ../lib/local/LandmarkDetector/include/LandmarkDetectorParameters.h
    ../lib/local/LandmarkDetector/include/LandmarkDetectorFunc.h
    ../lib/local/LandmarkDetector/include/LandmarkDetectorUtils.h
)


# Link required libraries to PyOpenFace binary

target_link_libraries(
    _openface PRIVATE
    ${LANDMARK_LIB}
    ${OpenCV_LIBS}
    ${OpenBLAS_LIB}
    dlib::dlib
)
if(${Boost_FOUND})
	target_link_libraries(_openface PRIVATE ${Boost_LIBRARIES})
else()
	target_link_libraries(_openface PRIVATE stdc++fs)
endif()
