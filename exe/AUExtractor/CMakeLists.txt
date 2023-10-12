# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
add_executable(AUExtractor AUExtractor.cpp base64.cpp)

find_package(cppzmq)

target_link_libraries(AUExtractor LandmarkDetector)
target_link_libraries(AUExtractor FaceAnalyser)
target_link_libraries(AUExtractor GazeAnalyser)
target_link_libraries(AUExtractor Utilities)
target_link_libraries(AUExtractor cppzmq)

install (TARGETS AUExtractor DESTINATION bin)