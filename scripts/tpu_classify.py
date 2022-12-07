#!/usr/bin/env python3
import sys, time
import numpy as np
import cv2
import io

import roslib
import rospy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Ros Messages
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from kidbright_ai_tpu.msg import tpu_object
from kidbright_ai_tpu.msg import tpu_objects

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

VERBOSE=False

class image_feature:

    def __init__(self, path):
        # topic where we publish
        self.font_path = "/home/pi/python/cascadia_font/CascadiaCode-Regular-VTT.ttf"
        self.font = ImageFont.truetype(self.font_path, 15)
        
        self.labels = read_label_file(path + '/output/labels.txt') 

        self.interpreter = make_interpreter(path + '/output/Classifier_best_val_accuracy_edgetpu.tflite')
        self.interpreter.allocate_tensors()
        
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        # self.bridge = CvBridge()
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)
        self.size = 320, 240
        
        rospy.init_node('image_class', anonymous=False)
  
    def callback(self, ros_data):
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        prepimg = image_np[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)
        draw = ImageDraw.Draw(prepimg)
        t1 = time.time()
        tpu_objects_msg = tpu_objects()
        size = common.input_size(self.interpreter)
        
        image = prepimg.convert('RGB').resize(size, Image.ANTIALIAS)
        common.set_input(self.interpreter, image)
        self.interpreter.invoke()
        
        out = classify.get_classes(self.interpreter, top_k=1, score_threshold=0.90)
        
        if out and len(out) == 1:
            if self.labels:
                target_id = out[0].id
                target_score = out[0].score
                target_label = self.labels[target_id]
                text = f"{target_label} {target_score:0.2f}"
                draw.text((10, 20), text, font=self.font, fill='green')
                
                tpu_object_m = tpu_object()
                tpu_object_m.cx = 0
                tpu_object_m.cy = 0
                tpu_object_m.width = 0
                tpu_object_m.height = 0
                tpu_object_m.label = target_label
                tpu_object_m.confident = target_score
                tpu_objects_msg.tpu_objects.append(tpu_object_m)
        
        t2 = time.time()
        fps = 1/(t2-t1)
        fps_str = 'FPS = %.2f' % fps
        draw.text((10,220), fps_str , fill='green')

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        #prepimg.save(fileIO,'jpeg')
        #msg.data = np.array(fileIO.getvalue()).tostring()
        #prepimg = prepimg.resize(self.size, Image.ANTIALIAS)
        open_cv_image = np.array(prepimg) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tostring()
        #msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)
        
        #self.subscriber.unregister()
    

if __name__ == '__main__':
    ic = image_feature(sys.argv[1])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    
