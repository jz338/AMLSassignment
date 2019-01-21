import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import math
from sklearn.svm import SVC
import csv
import sys

class face_detection:
   def __init__(self, img_dir):
      self.detector = dlib.get_frontal_face_detector()
      self.predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
      self.img_dir = img_dir

   def shape_to_np(self, shape, dtype="int"):
      # initialize the list of (x, y)-coordinates
      coords = np.zeros((shape.num_parts, 2), dtype=dtype)

      # loop over all facial landmarks and convert them
      # to a 2-tuple of (x, y)-coordinates
      for i in range(0, shape.num_parts):
         coords[i] = (shape.part(i).x, shape.part(i).y)

      # return the list of (x, y)-coordinates
      return coords

   def rect_to_bb(self, rect):
      # take a bounding predicted by dlib and convert it
      # to the format (x, y, w, h) as we would normally do
      # with OpenCV
      x = rect.left()
      y = rect.top()
      w = rect.right() - x
      h = rect.bottom() - y

      # return a tuple of (x, y, w, h)
      return (x, y, w, h)

   def process_landmarks(self, shape):
      xlist = []
      ylist = []
      for coord in shape:
         xlist.append(float(coord[0]))
         ylist.append(float(coord[1]))
      xmean = np.mean(xlist)
      ymean = np.mean(ylist)
      xcentral = [(x-xmean) for x in xlist]
      ycentral = [(y-ymean) for y in ylist]
      vectorised_landmarks = []
      for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
         vectorised_landmarks.append(w)
         vectorised_landmarks.append(z)
         meannp = np.asarray((ymean, xmean))
         coordnp = np.asarray((z,w))
         dist = np.linalg.norm(coordnp-meannp)
         vectorised_landmarks.append(dist)
         vectorised_landmarks.append((math.atan2(y, x)*360)/(2*math.pi))
      return vectorised_landmarks

   def run_dlib_shape(self, image):
      # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
      # load the input image, resize it, and convert it to grayscale
      resized_image = image.astype('uint8')

      gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
      gray = gray.astype('uint8')

      # detect faces in the grayscale image
      rects = self.detector(gray, 1)
      num_faces = len(rects)
      #print(num_faces)
      if num_faces == 0:
         return None, resized_image, None    

      face_areas = np.zeros((1, num_faces))
      face_shapes = np.zeros((136, num_faces), dtype=np.int64)

      # loop over the face detections
      for (i, rect) in enumerate(rects):
         # determine the facial landmarks for the face region, then
         # convert the facial landmark (x, y)-coordinates to a NumPy
         # array
         temp_shape = self.predictor(gray, rect)
         temp_shape = self.shape_to_np(temp_shape)

         # convert dlib's rectangle to a OpenCV-style bounding box
         # [i.e., (x, y, w, h)],
         #   (x, y, w, h) = face_utils.rect_to_bb(rect)
         (x, y, w, h) = self.rect_to_bb(rect)
         face_shapes[:, i] = np.reshape(temp_shape, [136])
         face_areas[0, i] = w * h
      # find largest face and keep
      dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])
      vectorised_landmarks = self.process_landmarks(dlibout)

      return dlibout, resized_image, vectorised_landmarks

   def extract_features_labels(self):
      """
      This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
      It also extract the gender label for each image.
      :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
      """
      image_paths = [os.path.join(self.img_dir, str(i)+'.png') for i in range(1,len(os.listdir(self.img_dir))+1)]
      target_size = None
      with open('attribute_list.csv') as csvfile:
         lines = csvfile.readlines()
      attrList = dict()
      for line in lines[2:]:
         attrs = line.split(',')
         attrList[attrs[0]] = {
            'hair_color' : int(attrs[1]),
            'eye_glasses' : int(attrs[2]),
            'smiling' : int(attrs[3]),
            'young' : int(attrs[4]),
            'human' : int(attrs[5])
         }
      attrCount = 0
      faceCount = 0
      cases = 0
      matches = 0
      if os.path.isdir(self.img_dir):
         all_features = {}
         vect_LMs = {}
         for img_path in image_paths:
            file_name= img_path.split('.')[0].split('/')[-1]
            print(file_name)
            # load image
            img = image.img_to_array(
               image.load_img(img_path,
                  target_size=target_size,
                  interpolation='bicubic'))
            features, _ , vect_lm = self.run_dlib_shape(img)
            cases += 1
            all_features[file_name] = features
            vect_LMs[file_name] = vect_lm

      #andmark_features = np.array(all_features)

      return all_features, vect_LMs, attrList


class SVC_trainer:
   def __init__(self, test_ratio=0.2, d_path="dataset"):
      self.test_ratio = test_ratio
      self.d_path = d_path
      self.datasize = len(os.listdir(self.d_path))
      self.clf = SVC(kernel='linear', probability=True, tol=1e-3)

   def rand_trainingset(self):
      dataidx = list(range(1, self.datasize+1))
      np.random.shuffle(dataidx)
      test_size = int(self.datasize*self.test_ratio)
      self.val_idx = dataidx[1:test_size+1]
      self.training_idx = dataidx[test_size+1:self.datasize+1]

   def prep_training_data(self):
      fd = face_detection("dataset")
      print("Preprocessing training_data....")
      self.lm_features, self.vect_LMs, self.attrList = fd.extract_features_labels()

   def assign_training_set(self, attr):
      training_data = []
      training_labels = []
      prediction_data = []
      prediction_labels = []
      self.val_files = []
      for i in self.training_idx:
         key = str(i)
         if self.lm_features[key] is not None:
            training_data.append(self.vect_LMs[key])
            training_labels.append(self.attrList[key][attr])
      for i in self.val_idx:
         key = str(i)
         if self.lm_features[key] is not None:
            self.val_files.append(key)
            prediction_data.append(self.vect_LMs[key])
            prediction_labels.append(self.attrList[key][attr])
      return training_data, training_labels, prediction_data, prediction_labels

   def trainingNvalidation(self, attr):
      td, tl, pd, pl = self.assign_training_set(attr)
      npar_td = np.array(td)
      npar_tl = np.array(tl)
      print("Training...")
      self.clf.fit(npar_td, npar_tl)
      npar_pd = np.array(pd)
      self.npar_pl = np.array(pl)
      print("Validating.....")
      self.pred_labels = self.clf.predict(npar_pd)
      self.pred_lin = self.clf.score(npar_pd, pl)
      print(self.pred_lin)
      print("linear SVM: {}".format(self.pred_lin))

   def prep_testing_data(self):
      t_fd = face_detection("testing_dataset")
      print("Preprocessing testing dataset....")
      self.t_lm_features, self.t_vect_LMs, self.t_attrList = t_fd.extract_features_labels()

   def testing(self, attr):
      testing_data = []
      testing_labels = []
      self.test_files = []
      self.test_idx = list(range(1, len(os.listdir("testing_dataset"))+1))
      for i in self.test_idx:
         key = str(i)
         if self.t_lm_features[key] is not None:
            self.test_files.append(key)
            testing_data.append(self.t_vect_LMs[key])
            testing_labels.append(self.t_attrList[key][attr])
      npar_testd = np.array(testing_data)
      self.npar_testl = np.array(testing_labels)
      print("Testing model.....")
      self.test_pred_labels = self.clf.predict(npar_testd)
      self.test_lin = self.clf.score(npar_testd, testing_labels)



   def write_csv(self, test_num):
      print("Writing CSVs...")
      writepath = os.path.join("lin_svc_result", "validation_"+test_num+".csv")
      with open(writepath, 'w') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         writer.writerow([self.pred_lin, ''])
         for idx, file in enumerate(self.val_files):
            file_name = file + '.png'
            writer.writerow([file_name, self.pred_labels[idx]])

      writepath = os.path.join("lin_svc_result", "test_"+test_num+".csv")
      with open(writepath, 'w') as csvfile:
         writer = csv.writer(csvfile, delimiter=',')
         writer.writerow([self.test_lin, ''])
         for idx, file in enumerate(self.test_files):
            file_name = file + '.png'
            writer.writerow([file_name, self.test_pred_labels[idx]])




def main():
   trainer = SVC_trainer()
   trainer.prep_training_data()
   trainer.prep_testing_data()
   trainer.rand_trainingset()
   trainer.trainingNvalidation("smiling")
   trainer.testing("smiling")
   trainer.write_csv("1")
   trainer.trainingNvalidation("young")
   trainer.testing("young")
   trainer.write_csv("2")
   trainer.trainingNvalidation("eye_glasses")
   trainer.testing("eye_glasses")
   trainer.write_csv("3")
   trainer.trainingNvalidation("human")
   trainer.testing("human")
   trainer.write_csv("4")
   #test()
   #fd = face_detection("dataset")
   #fd.extract_features_labels()
   #nr = noise_removal()
   #nr.attrList_validation(begin = 1, end = 51)

if __name__ == "__main__": 
   main()