import cv2
import time

start_time = time.time()

img = cv2.imread("doggy.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
print ("time elapsed: {:.4f}s".format(time.time() - start_time))
cv2.waitKey(0)
cv2.destroyAllWindows()


