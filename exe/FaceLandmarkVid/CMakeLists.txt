#TBB library
include_directories(${TBB_ROOT_DIR}/include)

add_executable(FaceLandmarkVid FaceLandmarkVid.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)		

target_link_libraries(FaceLandmarkVid LandmarkDetector)
target_link_libraries(FaceLandmarkVid FaceAnalyser)

if(${dlib_FOUND})
    target_link_libraries(FaceLandmarkVid ${dlib_LIBRARIES})
else()
    target_link_libraries(FaceLandmarkVid dlib)
endif()

target_link_libraries(FaceLandmarkVid ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FaceLandmarkVid DESTINATION bin)
