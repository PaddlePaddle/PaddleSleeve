#!/bin/bash

#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib ./build_enc/encrypt_model ./model.nb user_config.json
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../Paddle-Lite/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/lib ./build_enc/encrypt_model ./model.nb
