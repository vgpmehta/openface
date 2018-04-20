#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(FaceLandmarkVidMulti FaceLandmarkVidMulti.cpp)
target_link_libraries(FaceLandmarkVidMulti LandmarkDetector)
target_link_libraries(FaceLandmarkVidMulti GazeAnalyser)
target_link_libraries(FaceLandmarkVidMulti FaceAnalyser)
target_link_libraries(FaceLandmarkVidMulti Utilities)

install (TARGETS FaceLandmarkVidMulti DESTINATION bin)
