include_directories(${TBB_ROOT_DIR}/include)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    src/EmotionDetector.cpp
    src/InmindEmotionDetector.cpp
)

set(HEADER_FILES
    include/EmotionDetector.h
    include/InmindEmotionDetector.h
)
