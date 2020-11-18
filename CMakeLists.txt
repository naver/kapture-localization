
cmake_minimum_required(VERSION 3.0)

# this is NOT a c++ project
project( kapture_localization NONE )
include( ExternalProject )

# DOCKER BASE BUILD ############################################################
# set(DOCKER_BASE_IMAGE "nvidia/cudagl:10.0-devel-ubuntu18.04"
#     CACHE STRING "docker base image to start from")

# set(IMAGE_NAME_KAPTURE_INTERNAL kapture-internal
#     CACHE STRING "docker base image name")


add_custom_target(
    docker-kapture
    COMMAND     docker build
                -t naver/kapture
                -f kapture/Dockerfile kapture/
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

add_custom_target(
    docker-kapture-localization
    COMMAND     docker build
                -t naver/kapture-localization
                -f kapture-localization/Dockerfile .
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

add_dependencies(docker-kapture-localization docker-kapture)

#add_dependencies(${IMAGE_NAME_KAPTURE_INTERNAL} kapture-source kapture-internal-source kapture-models-source SfmFormat-source kapture-server-source eigen-source cereal-source ceres-solver-source colmap-source openmvg-source pycolmap-source)
#set( IMAGE_NAME_NO_USER ${IMAGE_NAME_KAPTURE_INTERNAL})
#add_subdirectory(shared/user)
