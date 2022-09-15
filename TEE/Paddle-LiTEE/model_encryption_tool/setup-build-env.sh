#!/bin/bash

git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite/ && git checkout v2.9.1

git apply ../patches/paddle-lite.patch
cp -r ../patches/lite.utils.rapidjson/ lite/utils/rapidjson
