
#LandmarkDetector library
include_directories(../../local/LandmarkDetector/include)

#Utilities library
include_directories(../../local/Utilities/include)

SET(SOURCE
    src/GazeEstimation.cpp
)

SET(HEADERS
    include/GazeEstimation.h
)

add_library( GazeAnalyser ${SOURCE} ${HEADERS})
add_library( OpenFace::GazeAnalyser ALIAS GazeAnalyser)

target_include_directories(GazeAnalyser PUBLIC ./include)
target_include_directories(GazeAnalyser PUBLIC ../../3rdParty/dlib/include)
target_include_directories(GazeAnalyser PRIVATE ${GAZEANALYSER_SOURCE_DIR}/include)
target_include_directories(GazeAnalyser PRIVATE ${TBB_ROOT_DIR}/include)
target_include_directories(GazeAnalyser PRIVATE ${BLAS_ROOT_DIR})
target_include_directories(GazeAnalyser PRIVATE ${BOOST_INCLUDE_DIR})
target_include_directories(GazeAnalyser PRIVATE ../../3rdParty/OpenBLAS/include)


target_link_libraries(GazeAnalyser INTERFACE dlib)
target_link_libraries(GazeAnalyser INTERFACE ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS GazeAnalyser DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
