cmake_minimum_required(VERSION 3.16)
project(pcb_detect_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# onnxruntime 압축을 푼 경로를 -DORT_HOME=... 으로 넘겨준다
if(NOT DEFINED ORT_HOME)
    message(FATAL_ERROR "-DORT_HOME=<onnxruntime 압축 푼 경로> 를 지정해주세요")
endif()

add_executable(detect_board main.cpp)
target_include_directories(detect_board PRIVATE ${ORT_HOME}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(detect_board PRIVATE ${ORT_HOME}/lib)
target_link_libraries(detect_board onnxruntime)
# 실행 파일이 라이브러리 경로를 기억하게 해서 LD_LIBRARY_PATH 없이 실행되게 함
set_target_properties(detect_board PROPERTIES BUILD_RPATH ${ORT_HOME}/lib)
