FROM osrf/ros:humble-desktop

# Install ping utility
RUN apt-get update && apt-get install -y \
    iputils-ping \
    ros-humble-rmw-cyclonedds-cpp \
    && apt-get clean