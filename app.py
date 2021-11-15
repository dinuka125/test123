import cv2
from deepface import DeepFace

img = "babakka.jfif"# read the first image file
img2 = "downgraaadi.jfif"# read the second image file

result = DeepFace.verify(img,img2, model='facenet')

print(result)
