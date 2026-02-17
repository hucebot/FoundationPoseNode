#!/bin/bash

docker rm -f foundationposev2
DIR=$(pwd)/
xhost +local:docker
docker run \
    --name foundationposev2 \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --net host \
    --ipc host \
    --pid host \
    --name foundationposev2 \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    -e ROS_DOMAIN_ID=1\
    -v $(pwd)/configs/:/xml_configs \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\
    -e CYCLONEDDS_URI=/xml_configs/cyclonedds.xml\
    foundationposev2:latest \
    bash -c "cd $DIR && bash"

# IsRunning=`docker ps -f name=foundationposev2 | grep -c "foundationposev2"`;
# if [ $IsRunning -eq "0" ]; then
#     echo "Docker image is not running. Starting it...";
#     xhost +local:docker
#     docker rm foundationposev2
#     docker run  \
#         --name foundationposev2  \
#         --gpus all \
#         -e DISPLAY=$DISPLAY \
#         -e NVIDIA_DRIVER_CAPABILITIES=all \
#         -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
#         -v /tmp/.X11-unix:/tmp/.X11-unix \
#         --env QT_X11_NO_MITSHM=1 \
#         --net host \
#         --ipc host \
#         --pid host \
#         --privileged \
#         -it \
#         -v $(pwd):/host_ws \
#         -v /dev:/dev \
#         -v /run/udev:/run/udev \
#         --device /dev/dri \
#         --device /dev/snd \
#         --device /dev/input \
#         --device /dev/bus/usb \
#         -e ROS_DOMAIN_ID=1\
#         -v $(pwd)/configs/:/xml_configs \
#         -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\
#         -e CYCLONEDDS_URI=/xml_configs/cyclonedds.xml\
#         -w /ros2_ws \
#         foundationposev2:latest
# else
#     echo "Docker image is already running. Opening new terminal...";
#     docker exec -ti foundationposev2 /bin/bash
# fi
