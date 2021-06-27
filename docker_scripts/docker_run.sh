#!/bin/bash

echo ""
echo "Running nn_experiments container"
echo ""

port=2600
name='nn_experiments'

while getopts p:n: option
do
    case "${option}"
        in
        p) port=${OPTARG};; 
        n) name=${OPTARG};;
    esac
done

echo "Using port: $port"
echo "Using container name: $name"

sudo docker run -it \
    -v /home/$USER/nn_experiments:/home/$USER/nn_experiments \
    -p $port:22 \
    --name $name \
    --gpus all \
    nn_experiments \
    /bin/bash
