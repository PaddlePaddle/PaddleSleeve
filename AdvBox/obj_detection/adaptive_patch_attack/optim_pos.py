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
The optimal patch position implementation.
Contains:
* Generate a batch of images with multiple patch candidate locations.
* Run object detection algorithm for the batch of images.
* Filter a few better patch locations.

Author: tianweijuan
"""

import cv2
img_cv = cv2.imread("./jpgs_320.jpg")
bnd_xmin, bnd_ymin, bnd_xmax, bnd_ymax = 274, 164, 589, 338

img = img_cv.copy()
for i in range(bnd_xmin, bnd_xmax-150, 5):
    for j in range(bnd_ymin, bnd_ymax-90, 5):
        img = img_cv.copy()
        img[j:j+90, i:i+150, :] = 0.
        cv2.imwrite("./optim_cand_toy/"+ str(i)+"_"+str(j)+".png", img)

img = img_cv.copy()
for i in range(bnd_xmin, bnd_xmax-150, 5):
    for j in range(bnd_ymax-90, bnd_ymin, -5):
        img = img_cv.copy()
        img[j:j+90, i:i+150, :] = 0.
        cv2.imwrite("./optim_cand_toy/"+ str(i)+"_"+str(j)+".png", img)

img = img_cv.copy()
for i in range(bnd_xmax-150, bnd_xmin, -5):
    for j in range(bnd_ymax-90, bnd_ymin, -5):
        img = img_cv.copy()
        img[j:j+90, i:i+150, :] = 0.
        cv2.imwrite("./optim_cand_toy/"+ str(i)+"_"+str(j)+".png", img)

img = img_cv.copy()

for i in range(bnd_xmax-150, bnd_xmin, -5):
    for j in range(bnd_ymin, bnd_ymax-90, 5):
        img = img_cv.copy()
        img[j:j+90, i:i+150, :] = 0.
        cv2.imwrite("./optim_cand_toy/"+ str(i)+"_"+str(j)+".png", img)




      

