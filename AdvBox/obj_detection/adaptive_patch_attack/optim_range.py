# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The optimal patch size implementation.
Contains:
* Generate a batch of images with multiple patch candidate sizes.
* Run object detection algorithm for the batch of images.
* Filter a few better patch sizes.

Author: tianweijuan
"""
import cv2

img_cv = cv2.imread("./jpgs_320.jpg")
bnd_xmin, bnd_ymin = 424, 223

for wid in range(150, 165, 5):
    for hig in range(80, 115, 5):
        bnd_xmax = bnd_xmin + wid
        bnd_ymax = bnd_ymin + hig
        img = img_cv.copy()
        img[bnd_ymin:bnd_ymax, bnd_xmin:bnd_xmax, :] = 0.
        
        cv2.imwrite("./optim_range_toy/"+ str(wid)+"_"+str(hig)+".png", img)




      

