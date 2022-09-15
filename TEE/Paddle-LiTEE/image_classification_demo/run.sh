#!/bin/bash

PADDLE_LITE_DIR=/root/shared/build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8/cxx/

# run
CONFIG_DIR=/root/shared/tool/ LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/lib ./build/image_classification_demo /root/shared/tool/model_tee.nb ./labels/synset_words.txt ./images/tabby_cat.jpg ./result.jpg
