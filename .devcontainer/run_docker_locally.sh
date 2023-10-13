#!/usr/bin/env bash

image_name=${1:-nimrod-paperspace}  
docker run --env-file env-file -p 8888:8888 -it --rm ${image_name} /bin/bash
