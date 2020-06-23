import numpy as np
import tensorflow as tf
import cv2
import time
from keras import applications
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
from mobileNetFeatureExtract import intermediate_output
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
sns.set(style ="whitegrid") 


model = applications.mobilenet.MobileNet()
feature = []
feature2 = []
names = []
pf = []
classN = []
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

    
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
     
        image_np_expanded = np.expand_dims(image, axis=0)
   
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = 'ssd_mobilenet/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)

    cap = cv2.VideoCapture('TEST.mp4')

    out = cv2.VideoWriter('output.mp4', -1, 20.0, (1280, 720))
 
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
     
        personNumber=0
        personIndividual=0
        names = []
        pf = []

        for i in range(len(boxes)):
            
            if classes[i] == 1 and scores[i] > 0.3:  # Class 1 represents human

                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                roi = img[box[0]:box[1], box[1]:box[3]]
                cv2.imwrite("roi.png", roi)
                personIndividual+=1
                cv2.putText(img,str(personIndividual),(box[1],box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                personNumber+=1    
                value = intermediate_output(model,img)
                classN.append(personNumber)
                feature.append(value)
                n = ['Person' + str(i)]*1000
                names.extend(n)
                pf.extend(np.squeeze(value))

        cv2.putText(img,str(personNumber),(10,500),font,4,(255,255,255),2,cv2.LINE_AA)     

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
    x = np.squeeze(np.array(feature))
    y = np.squeeze(np.array(classN))
    print(x.shape)
    print(y.shape)

    np.savetxt('2darray.csv', x, delimiter=',', fmt='%d')
    np.savetxt('21darray.csv', y, delimiter=',', fmt='%s')