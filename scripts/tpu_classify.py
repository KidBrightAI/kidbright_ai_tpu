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
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


VERBOSE=False

class image_feature:

    def __init__(self, path):
        # topic where we publish
        self.labels = read_label_file(path + '/labels.txt') 
        self.interpreter = make_interpreter(path + '/model_edgetpu.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        _, self.input_height, self.input_width, _ = self.input_details['shape']
        self.output_details = self.interpreter.get_output_details()[0]
        
        #Init Node
        rospy.init_node('image_class', anonymous=False)

        # To publish topic
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)
        self.size = 320, 240
        
  
    def preprocess(self, img):
        
        img = img.astype(np.float32)
        img = img / 255.
        img = img - 0.5
        img = img * 2.
        img = img[:, :, ::-1]
        
        return img

    def callback(self, ros_data):
        t1 = time.time()

        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 

        # resize image
        input_np = cv2.resize(image_np.copy(), (self.input_width, self.input_height))
        input_np = np.expand_dims(input_np, 0)

        self.interpreter.set_tensor(self.input_details["index"], input_np)
        self.interpreter.invoke()
        
        out = classify.get_classes(self.interpreter, top_k=1)

        tpu_objects_msg = tpu_objects()
        
        if out and len(out) == 1:
            if self.labels:
                target_id = out[0].id
                target_score = out[0].score
                target_label = self.labels[target_id]
                text = f"{target_label} {target_score:0.2f}"
                image_np = cv2.putText(image_np, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
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
        image_np = cv2.putText(image_np, fps_str, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"

        open_cv_image = image_np[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tostring()

        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)
        
        # #self.subscriber.unregister()
    

if __name__ == '__main__':
    ic = image_feature(sys.argv[1])
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    
