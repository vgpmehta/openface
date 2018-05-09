add_executable(FaceLandmarkVid FaceLandmarkVid.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

target_link_libraries(FaceLandmarkVid LandmarkDetector)
target_link_libraries(FaceLandmarkVid FaceAnalyser)
target_link_libraries(FaceLandmarkVid GazeAnalyser)
target_link_libraries(FaceLandmarkVid Utilities)

install (TARGETS FaceLandmarkVid DESTINATION bin)
