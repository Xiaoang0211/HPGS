# SPDX-FileCopyrightText: 2021-2023 Smart Robotics Lab, Imperial College London, Technical University of Munich
# SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.24)

find_package(GLUT)
find_package(OpenGL)
find_package(PCL COMPONENTS io)
find_package(PkgConfig) # For OpenNI2
if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENNI2 libopenni2)
else()
    set(OPENNI2_FOUND OFF)
endif()

# we must compile with ros2
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)

# Define the absolute path to Open3D
get_filename_component(PROJ_ROOT_DIR "${PROJECT_SOURCE_DIR}" ABSOLUTE)
set(OPEN3D_DIR "${PROJ_ROOT_DIR}/third_party/open3d")
list(APPEND CMAKE_PREFIX_PATH ${OPEN3D_DIR})
find_package(Open3D REQUIRED)

set(LIB_NAME reader)
add_library(${LIB_NAME} STATIC
  "src/config.cpp"
  "src/reader.cpp"
  "src/reader_base.cpp"
  "src/reader_replica.cpp"
  "src/reader_scannetpp.cpp"
  "src/reader_tum.cpp"
)
target_include_directories(${LIB_NAME} PUBLIC include)
target_link_libraries(${LIB_NAME} PUBLIC SRL::Supereight2 )
if(OPENMP_FOUND)
    target_link_libraries(${LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
if(IPO_SUPPORTED)
  set_property(TARGET ${LIB_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
if(PCL_FOUND)
  # These have to be added globaly for compatibility with PCL 1.8.1 on Ubuntu 18.04.
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  target_link_libraries(${LIB_NAME} PUBLIC ${PCL_LIBRARIES})
  target_compile_definitions(${LIB_NAME} PUBLIC "-DSE_PCL")
endif()
if(OPENNI2_FOUND)
    target_include_directories(${LIB_NAME} PRIVATE ${OPENNI2_INCLUDE_DIRS})
    target_link_libraries(${LIB_NAME} PRIVATE ${OPENNI2_LIBRARIES})
    target_compile_definitions(${LIB_NAME} PRIVATE "-DSE_OPENNI2")
endif()


set(MAIN_ROS2 "${CMAKE_CURRENT_SOURCE_DIR}/src/listener.cpp")
set(EXE_NAME_ROS2 "hpgs_ros2")
add_executable(${EXE_NAME_ROS2} ${MAIN_ROS2} "src/gui.cpp")
target_include_directories(${EXE_NAME_ROS2} BEFORE PRIVATE include)
target_link_libraries(${EXE_NAME_ROS2} SRL::Supereight2 ${LIB_NAME})
target_link_libraries(${EXE_NAME_ROS2} Open3D::Open3D)
# link ros2 libraries
ament_target_dependencies(${EXE_NAME_ROS2}
  rclcpp
  cv_bridge
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  message_filters
  image_transport
)

set_target_properties(${EXE_NAME_ROS2} PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../../../src/HPGS_ROS2/third_party/Open3D/build/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# for the executable with ros2
if(OPENMP_FOUND)
  target_link_libraries(${EXE_NAME_ROS2} OpenMP::OpenMP_CXX)
endif()

if(IPO_SUPPORTED)
  set_property(TARGET ${EXE_NAME_ROS2} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_compile_definitions(${EXE_NAME_ROS2} PRIVATE _FILE_OFFSET_BITS=64)

if(GLUT_FOUND)
  target_link_libraries(${EXE_NAME_ROS2}
      ${GLUT_LIBRARIES}
      ${OPENGL_LIBRARIES}
  )
  target_compile_definitions(${EXE_NAME_ROS2} PRIVATE "-DSE_GLUT")

  if(NOT APPLE)
    target_link_libraries(${EXE_NAME_ROS2} GL)
  endif()
endif()

install(TARGETS ${EXE_NAME} ${EXE_NAME_ROS2}
  DESTINATION lib/${PROJECT_NAME}
)