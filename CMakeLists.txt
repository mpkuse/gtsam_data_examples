cmake_minimum_required(VERSION 3.2)
project(gtsam_tutorial)


### Libraries ###
find_package(GTSAM REQUIRED) # Uses installed package
include_directories(${GTSAM_INCLUDE_DIR})
link_directories(${GTSAM_LIBRARY_DIRS})

find_package(OpenCV 4.5 REQUIRED )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${EIGEN3_INCLUDE_DIRS} )

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



### Utils Library ###
add_library(${PROJECT_NAME}_lib
    src/utils/CvPlottingUtils.cc 
)
target_link_libraries(${PROJECT_NAME}_lib gtsam ${OpenCV_LIBRARIES} gtest  )


### SRC ###
add_executable( hello-gtsam src/hello-gtsam.cc )
target_link_libraries( hello-gtsam  gtsam )


add_executable( hello-opencv src/hello-opencv.cc )
target_link_libraries( hello-opencv  ${OpenCV_LIBRARIES} ${PROJECT_NAME}_lib gtest )

