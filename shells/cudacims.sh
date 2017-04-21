#!/usr/bin/env bash
# needed for running cuda on cims machines
# https://cims.nyu.edu/webapps/content/systems/resources/computeservers/cuda

os=${1:-centos}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ $env = "centos" ]
then
	module load mpi/mpich-x86_64

    cp -r /usr/local/cuda/samples ~/samples

    cd ~/samples

    make

    cd bin/x86_64/linux/release

    ./deviceQuery

    ./bandwidthTest

    cd $DIR

elif [ $env = "ubuntu" ]
then
	module load mpich2

    cp -r /usr/local/pkg/cuda/current/sdk ~/nvidia_sdk

    cd ~/nvidia_sdk

    make

    cd bin/linux/release

    ./deviceQuery

    ./bandwidthTest

     cd $DIR
fi