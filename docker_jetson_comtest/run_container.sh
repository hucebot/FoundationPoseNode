#!/bin/bash
# Minimal ROS2 Jazzy container for connectivity testing on the Jetson AGX Thor.
# No GPU flags: this only exercises networking and DDS, so it runs even before the
# NVIDIA container runtime is set up. Run it from the repository root.
IsRunning=`docker ps -f name=fp_comtest | grep -c "fp_comtest"`;
DIR=$(pwd)/
if [ $IsRunning -eq "0" ]; then
    docker rm -f fp_comtest
    # Applied on the host: the container shares the host network namespace.
    sudo sysctl -w net.ipv4.ipfrag_time=3 > /dev/null 2>&1 || true
    sudo sysctl -w net.ipv4.ipfrag_high_thresh=134217728 > /dev/null 2>&1 || true
    echo "Host interfaces (for setting NetworkInterface in configs/cyclonedds.xml):"
    ip -brief addr
    docker run \
        --name fp_comtest \
        -it \
        --net host \
        --ipc host \
        --pid host \
        -v $DIR:$DIR \
        -e ROS_DOMAIN_ID=2 \
        -v $(pwd)/configs/:/xml_configs \
        -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
        -e CYCLONEDDS_URI=/xml_configs/cyclonedds.xml \
        fp_comtest:latest \
        bash -c "cd $DIR && bash"
else
    echo "Docker image is already running. Opening new terminal...";
    docker exec -ti fp_comtest bash -c "cd $DIR && bash"
fi
