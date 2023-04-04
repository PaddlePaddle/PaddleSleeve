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




      

