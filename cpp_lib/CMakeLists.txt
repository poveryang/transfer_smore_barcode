cmake_minimum_required(VERSION 3.16)

project(roi_projector_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(roi_projector
  roi_projector.cpp
)

target_include_directories(roi_projector
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(roi_projector_test
  test_roi_projector.cpp
)

target_link_libraries(roi_projector_test
  PRIVATE
    roi_projector
)
