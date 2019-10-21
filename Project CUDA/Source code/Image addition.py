import cv2
import time

start_time = time.time()
img1 = cv2.imread("doggy.jpg", 0)
img2 = cv2.imread("Illuminati.png", 0)

img3 = img1 + img2

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.imshow("Added image", img3)
print ("time elapsed: {:.4f}s".format(time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()


