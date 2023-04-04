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
The code for video to images.

Author: tianweijuan
"""

import cv2
import os

def video2imgs(videoPath, imgPath):
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)             # If the destination folder does not exist, create
    cap = cv2.VideoCapture(videoPath)    # obtain video
    judge = cap.isOpened()               # Determine if it can be opened successfully
    print(judge)
    fps = cap.get(cv2.CAP_PROP_FPS)      # Frame rate, how many pictures per second the video shows
    print('fps:',fps)

    frames = 1                           # used for counting all frames
    count = 0                            # used for counting the number of saved images

    while(judge):
        
        #Read each image flag indicates whether the read is successful or not, frame is the image
        flag, frame = cap.read()         
        if not flag:
            print(flag)
            print("Process finished!")
            break
        else:
            if frames % 3 == 0: 
                imgname = 'jpgs_' + str(count).rjust(3,'0') + ".jpg"
                newPath = imgPath + imgname
                print(imgname)
                cv2.imwrite(newPath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
                if count > 930:
                    break
        frames += 1
    cap.release()
    print("Total %d pictures"%(count-1))
video2imgs('./patch_truck.mp4','./truck_toy/')
