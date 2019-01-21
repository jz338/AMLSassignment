import numpy as np
import cv2
import matplotlib.pyplot as plt
import time 
import csv
import os

class face_detection:
   def __init__(self, scaleFactor = 1.1):
      self.scaleFactor = scaleFactor
      self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

   def detect_faces(self, coloured_img):
      img_copy = np.copy(coloured_img)
      gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
      faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scaleFactor, minNeighbors=3)
      for (x, y, w, h) in faces:
         cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

      cv2.imshow('Test Imag', img_copy)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      if len(faces) > 0:
         
         return True
      else:
         return False

class noise_removal:
   def __init__(self, d_path='dataset', img_type='.png'):
      self.d_path = d_path
      self.img_type = img_type
      self.fd = face_detection()

   def list_img_files(self):
      return os.listdir(self.d_path)

   def has_face(self, img_file):
      image = cv2.imread(os.path.join(self.d_path, img_file))
      return self.fd.detect_faces(image)

   def attrList_validation(self, begin = 1, end = 5001):
      data = csv_process()
      attrCount = 0
      faceCount = 0
      cases = 0
      matches = 0
      for i in range(begin, end):
         print(i)
         cases += 1
         img_file = str(i) + self.img_type
         if self.has_face(img_file):
            faceCount += 1
            if int(data[i - 1][1]) != -1:
               attrCount += 1
               matches += 1
               #print([1,1])
            else:
               pass
               #print([0,1])
         else:
            if int(data[i - 1][1]) == -1:
               matches += 1
               #print([0,0])

            else:
               attrCount += 1
               #print([0,0])

      print([attrCount, faceCount, cases, matches])



def csv_process():
   with open('attribute_list.csv') as csvfile:
      data = list(csv.reader(csvfile))

   del data[0]
   del data[0]

   return data

def main():
   #fd = face_detection()
   #test_img = cv2.imread('dataset/1.png')
   #fd.detect_faces(test_img)

   #attribute_list = csv_process()

   #p#rint(attribute_list[0])
   nr = noise_removal()
   #img_files = nr.list_img_files()
   nr.attrList_validation(begin = 1, end = 51)

if __name__ == "__main__": main()