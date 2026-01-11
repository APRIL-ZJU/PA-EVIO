cmake_minimum_required(VERSION 2.8.3)
project(ts_ros)

# To be consistent with the configuration in esvo_core
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
# set(CMAKE_CXX_FLAGS "-O3") # Use target_compile_options instead

catkin_package(CATKIN_DEPENDS roscpp dvs_msgs
  INCLUDE_DIRS include
  LIBRARIES ts_ros_lib
)                                          

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})

# --- Library Definition (Library First Strategy) ---
# Create the library first so executables can link against it
# This avoids compiling the same source files multiple times

# List of source files for the library
set(LIBRARY_SOURCES
    src/AdaptiveTSurface.cpp
    src/TimeSurface.cpp
)

# Use standard add_library to maintain control over shared/static and specific properties
cs_add_library(ts_ros_lib SHARED ${LIBRARY_SOURCES})

# Target properties
target_link_libraries(ts_ros_lib
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
)
target_include_directories(ts_ros_lib PUBLIC include/)
# target_compile_features(ts_ros_lib PUBLIC cxx_std_14) # Already set globally
target_compile_options(ts_ros_lib PRIVATE -O3) # Target specific optimization

# --- Executables ---

# make the executable
cs_add_executable(time_surface
    src/TimeSurface_node.cpp
)
target_link_libraries(time_surface
   ts_ros_lib
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

# make the executable
cs_add_executable(adaptive_time_surface
    src/ATimeSurface_node.cpp
)
target_link_libraries(adaptive_time_surface
   ts_ros_lib
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

cs_install()

# Install the library and headers
install(TARGETS ts_ros_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
