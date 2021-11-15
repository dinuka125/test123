import cv2
from deepface import DeepFace

img = cv2.imread("image1.jpg")# read the first image file
img2 = cv2.imread("image2.jpg")# read the second image file

result = DeepFace.verify(img,img2, model='facenet')

print(result)
