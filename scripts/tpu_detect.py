#!/usr/bin/env python3
import sys, time
import numpy as np
import cv2
import io
import json
import os

import roslib
import rospy

try:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

#YOLO
from decoder import YoloDecoder
from box import to_minmax

VERBOSE=False

class image_feature:

    def __init__(self, path, threshold = 0.5):
        '''Initialize ros publisher, ros subscriber'''
        # path = "/home/pi/kbai-server/inferences/yolo"
        self.threshold = threshold
        self.labels = self.read_label_file(path + '/labels.txt')
        
        project_json = self.read_json_file(path + "/project.json")
        self.anchors = project_json["project"]["project"]["anchors"]
        try:
            self.interpreter = make_interpreter(path + '/model_edgetpu.tflite')
            self.mode = "CORAL"
        except:
            self.interpreter = tflite.Interpreter(path + '/model_edgetpu.tflite')
            self.mode = "LEGACY"
            
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        _, self.input_height, self.input_width, _ = self.input_details['shape']
        self.output_details = self.interpreter.get_output_details()[0]
        self.output_scale = self.output_details["quantization_parameters"]["scales"][0]
        self.output_zero_points = self.output_details["quantization_parameters"]["zero_points"][0]
        
        #YOLO 
        if self.anchors:
            self.decoder = YoloDecoder(self.anchors)
        
        #Init Node
        rospy.init_node('image_feature', anonymous=False)

        #Publish
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)
        #Subscribe
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)
    
    def read_json_file(self, file):
        if os.path.exists(file):
            with open(file) as f:
                return json.load(f)

    def read_label_file(self, file_path):
        with open(file_path,'r') as f:
            line = f.readlines()
            return line[0].split(",")
        return False
    
    def bbox_to_xy(self,boxes,w,h):
        minmax_boxes = to_minmax(boxes)
        minmax_boxes[:,0] *= w
        minmax_boxes[:,2] *= w
        minmax_boxes[:,1] *= h
        minmax_boxes[:,3] *= h
        return minmax_boxes.astype(int)

    def callback(self, ros_data):
        t1 = time.time()
        
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 

        input_np = cv2.resize(image_np.copy(), (self.input_width, self.input_height)) #shape = 224,320,3
        input_np = np.expand_dims(input_np, 0) #shape 
        self.interpreter.set_tensor(self.input_details["index"], input_np)
        self.interpreter.invoke()
        
        #output_data = self.interpreter.tensor(self.output_details['index'])() #shape = 1, 7, 10, 35
        
        netout = self.interpreter.get_tensor(self.output_details['index']).astype(np.float32)
        netout = (netout - self.output_zero_points) * self.output_scale
        netout = netout.reshape(7, 10, 5, 7)
        
        boxes, probs = self.decoder.run(netout, self.threshold)

        tpu_objects_msg = tpu_objects()
        if len(boxes) > 0:
            boxes = self.bbox_to_xy(boxes,image_np.shape[1],image_np.shape[0])
            if self.labels:
                for box, classes in zip(boxes, probs):
                    x1, y1, x2, y2 = box
                    target_class_index = np.argmax(classes)
                    cv2.rectangle(image_np, (x1,y1), (x2,y2), (0,255,0), 3)
                    cv2.putText(image_np, 
                        "{}:  {:.2f}".format(self.labels[target_class_index], classes[target_class_index]), 
                        (x1, y1 - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.002 * image_np.shape[0], 
                        (255,0,0), 
                        1)
                    tpu_object_m = tpu_object()
                    tpu_object_m.cx = (x2 - x1) / 2
                    tpu_object_m.cy = (y2 - y1) / 2
                    tpu_object_m.width = x2 - x1
                    tpu_object_m.height = y2 - y1
                    tpu_object_m.label = self.labels[target_class_index]
                    tpu_object_m.confident = classes[target_class_index]
                    tpu_objects_msg.tpu_objects.append(tpu_object_m)
                    
        t2 = time.time()
        fps = 1/(t2-t1)
        fps_str = 'FPS = %.2f' % fps
        image_np = cv2.putText(image_np, fps_str, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)

        # #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        open_cv_image = image_np[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tobytes()
        
        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)

def main(path):
    '''Initializes and cleanup ros node'''
    ic = image_feature(path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])
