#TBB library
include_directories(${TBB_ROOT_DIR}/include)

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/Face_utils.cpp
    src/FaceAnalyser.cpp
    src/SVM_dynamic_lin.cpp
    src/SVM_static_lin.cpp
    src/SVR_dynamic_lin_regressors.cpp
    src/SVR_static_lin_regressors.cpp
    src/GazeEstimation.cpp
)

SET(HEADERS
    include/Face_utils.h	
    include/FaceAnalyser.h
    include/SVM_dynamic_lin.h
    include/SVM_static_lin.h
    include/SVR_dynamic_lin_regressors.h
    include/SVR_static_lin_regressors.h
    include/GazeEstimation.h
)

include_directories(./include)
include_directories(${FACEANALYSER_SOURCE_DIR}/include)
include_directories(../LandmarkDetector/include)

if(WIN32)
    add_library( FaceAnalyser ${SOURCE} ${HEADERS} )
else()
    add_library( FaceAnalyser SHARED ${SOURCE} ${HEADERS} )
    target_link_libraries( FaceAnalyser PUBLIC dlib::dlib-shared)
endif()

target_link_libraries( FaceAnalyser PUBLIC ${OpenCV_LIBS} ${TBB_LIBRARIES} ${Boost_LIBRARIES})

set_property(TARGET FaceAnalyser PROPERTY VERSION ${OpenFace_VERSION})

install (TARGETS FaceAnalyser
    EXPORT OpenFaceTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install (FILES ${HEADERS} DESTINATION include/OpenFace)
