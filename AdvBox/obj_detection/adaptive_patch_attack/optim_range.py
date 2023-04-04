import cv2


#img_cv = cv2.imread("1_229.jpg")
#bnd_xmin, bnd_ymin, bnd_xmax, bnd_ymax = 384, 410, 887, 627
img_cv = cv2.imread("./jpgs_320.jpg")
bnd_xmin, bnd_ymin = 424, 223

for wid in range(150, 165, 5):
    for hig in range(80, 115, 5):
        bnd_xmax = bnd_xmin + wid
        bnd_ymax = bnd_ymin + hig
        img = img_cv.copy()
        img[bnd_ymin:bnd_ymax, bnd_xmin:bnd_xmax, :] = 0.
        
        cv2.imwrite("./optim_range_toy/"+ str(wid)+"_"+str(hig)+".png", img)




      

