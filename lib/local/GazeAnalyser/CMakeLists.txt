
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

target_include_directories(GazeAnalyser PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/OpenFace>)

target_include_directories(GazeAnalyser PRIVATE ${GAZEANALYSER_SOURCE_DIR}/include)
target_include_directories(GazeAnalyser PRIVATE ${TBB_ROOT_DIR}/include)
target_include_directories(GazeAnalyser PUBLIC ${Boost_INCLUDE_DIRS})
target_include_directories(GazeAnalyser PUBLIC ${Boost_INCLUDE_DIRS}/boost)
target_include_directories(GazeAnalyser PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(GazeAnalyser INTERFACE ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

if(${BLAS_FOUND})
    target_include_directories(GazeAnalyser PRIVATE ${BLAS_ROOT_DIR})
else()
    target_include_directories(GazeAnalyser PRIVATE ../../3rdParty/OpenBLAS/include)
endif()

if(${dlib_FOUND})
    target_link_libraries(GazeAnalyser INTERFACE ${dlib_LIBRARIES})
    target_include_directories(GazeAnalyser PUBLIC ${dlib_INCLUDE_DIRS})
else()
    target_link_libraries(GazeAnalyser INTERFACE dlib)
endif()

install (TARGETS GazeAnalyser EXPORT OpenFaceTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
