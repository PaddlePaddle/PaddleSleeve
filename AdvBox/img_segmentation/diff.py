import cv2
import matplotlib.pyplot as plt
plt.figure()
adversarial_img = cv2.imread('./output/adv_P0024.jpg')
original_img = cv2.imread('./dataloader/P0024.jpg')
difference = adversarial_img - original_img
print ("diff shape: ", difference.shape)
# (-1,1)  -> (0,1)
difference = difference / abs(difference).max() / 2.0 + 0.5
plt.imshow(difference, cmap=plt.cm.gray)
plt.show()
