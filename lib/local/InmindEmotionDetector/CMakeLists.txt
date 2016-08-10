set(CMAKE_CXX_STANDARD 11)

set(SOURCE
    src/EmotionDetector.cpp
    src/InmindEmotionDetector.cpp
)

set(HEADERS
    include/EmotionDetector.h
    include/InmindEmotionDetector.h
)

include_directories(./include)
include_directories(../LandmarkDetector/include)
include_directories(../FaceAnalyser/include)

add_library( InmindEmotionDetector ${SOURCE} ${HEADERS})

install (TARGETS InmindEmotionDetector DESTINATION bin)
install (FILES HEADERS DESTINATION include)
