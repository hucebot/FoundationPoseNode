#!/bin/bash
# Jetson AGX Thor (JetPack 7.2 / L4T 39.2.0) variant of docker/run_container.sh.
# Run it from the repository root, the same way as the x86 script.
IsRunning=`docker ps -f name=foundationposev2_jetson | grep -c "foundationposev2_jetson"`;
DIR=$(pwd)/
if [ $IsRunning -eq "0" ]; then
    docker rm -f foundationposev2_jetson
    xhost +local:docker
    # Applied on the host: the container shares the host network namespace.
    # sudo sysctl -w net.ipv4.ipfrag_time=3 > /dev/null 2>&1 || true
    # sudo sysctl -w net.ipv4.ipfrag_high_thresh=134217728 > /dev/null 2>&1 || true
    docker run \
        --name foundationposev2_jetson \
        --runtime nvidia \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -it \
        --net host \
        --ipc host \
        --pid host \
        --privileged \
        -v /dev:/dev \
        -v /run/udev:/run/udev:ro \
        --device /dev/bus/usb \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v $DIR:$DIR \
        -v /home:/home \
        -v /mnt:/mnt \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /tmp:/tmp \
        -e DISPLAY=${DISPLAY} \
        -e GIT_INDEX_FILE \
        -e ROS_DOMAIN_ID=2\
        -v $(pwd)/configs/:/xml_configs \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\
        -e CYCLONEDDS_URI=/xml_configs/cyclonedds.xml\
        foundationposev2_jetson:latest \
        bash -c "cd $DIR && bash"
else
    echo "Docker image is already running. Opening new terminal...";
    docker exec -ti foundationposev2_jetson bash -c "cd $DIR && bash"
fi
