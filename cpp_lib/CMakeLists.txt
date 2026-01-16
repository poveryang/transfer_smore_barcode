cmake_minimum_required(VERSION 3.16)

project(roi_projector_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(ROI_PROJECTOR_BUILD_TEST "Build roi_projector_test executable" ON)

add_library(roi_projector SHARED
  roi_projector.cpp
)

target_include_directories(roi_projector
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ROI_PROJECTOR_BUILD_TEST)
  add_executable(roi_projector_test
    test_roi_projector.cpp
  )

  target_link_libraries(roi_projector_test
    PRIVATE
      roi_projector
  )
endif()

install(TARGETS roi_projector
  EXPORT roi_projectorTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(ROI_PROJECTOR_BUILD_TEST)
  install(TARGETS roi_projector_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/roi_projector.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
