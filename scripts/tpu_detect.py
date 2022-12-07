#!/usr/bin/env python3
import sys, time
import numpy as np
import cv2
import io

import roslib
import rospy

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from kidbright_ai_tpu.msg import tpu_object
from kidbright_ai_tpu.msg import tpu_objects

VERBOSE=False

class image_feature:

    def __init__(self, path):
        '''Initialize ros publisher, ros subscriber'''
        self.font_path = "/home/pi/python/cascadia_font/CascadiaCode-Regular-VTT.ttf"
        self.font = ImageFont.truetype(self.font_path, 15)
        
        #self.engine = DetectionEngine('/home/pi/customModels/output_tflite_graph_edgetpu-3.tflite')
        #self.engine = DetectionEngine('/home/pi/customModels/output_tflite_ssd_v2_corrected_graph_edgetpu-1000.tflite')
        #self.engine = DetectionEngine(path + '/model.tflite')
        self.labels = read_label_file(path + '/labels.txt') 
        self.interpreter = make_interpreter(path + '/model.tflite')
        self.interpreter.allocate_tensors()
        #self.engine = DetectionEngine('/home/pi/customModels/exampleRobot/output_tflite_graph_edgetpu.tflite')
        
        #self.labels = self.ReadLabelFile('/home/pi/customModels/totoro_label.txt') 

        #self.engine = DetectionEngine('/home/pi/TPU-MobilenetSSD/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        #self.labels = self.ReadLabelFile('/home/pi/TPU-MobilenetSSD/coco_labels.txt') 
        #self.labels = self.ReadLabelFile(path + '/labels.txt') 
        # self.labels = self.ReadLabelFile('/home/pi/customModels/exampleRobot/exampleRobot_labels.txt') 
        
        
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        # self.bridge = CvBridge()
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.vel_msg = Twist()
        rospy.init_node('image_feature', anonymous=False)


    def ReadLabelFile(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    def getObjectFeatures(self, box):
        width = box[2]-box[0]
        height = box[3]-box[1]
        area = width*height
        c_x = box[0] + width/2
        c_y = box[3] + height/2
    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''

        #fileIO = io.BytesIO()
        #### direct conversion to CV2 ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        #### Feature detectors using CV2 #### 
        # "","Grid","Pyramid" + 
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        prepimg = image_np[:, :, ::-1].copy()
        prepimg = Image.fromarray(prepimg)
        draw = ImageDraw.Draw(prepimg)
        #print("Hello detect !!!!")
        t1 = time.time()
        size = common.input_size(self.interpreter)
        
        image = prepimg.convert('RGB')
        _, scale = common.set_resized_input(self.interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
        self.interpreter.invoke()
        out = detect.get_objects(self.interpreter, 0.6, scale)
        tpu_objects_msg = tpu_objects()
        #print(out)
        if out:
            for obj in out:
                #print ('-----------------------------------------')
                #if labels:
                #    print(labels[obj.label_id])
                #print ('score = ', obj.score)
                bbox = obj.bbox
                #print ('box = ', box)
                # Draw a rectangle.
                
                width = bbox.xmax - bbox.xmin
                height = bbox.ymax - bbox.ymin
                area = width*height
                c_x = bbox.xmin + width/2
                c_y = bbox.ymin + height/2

                draw.ellipse((c_x-5, c_y-5, c_x+5, c_y+5), fill = 'blue', outline ='blue')
                if self.labels:
                    vbal = f"{self.labels[obj.id]} {obj.score:0.2f} {c_x:.2f} {area:.2f}"
                    #vbal = f"{self.labels[obj.label_id]} {box[0]} {box[1]} {box[2]} {box[3]}"
                    
                    
                    draw.text((bbox.xmin, bbox.xmin), vbal, font=self.font, fill='green')
                    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='green')
                        
                    tpu_object_m = tpu_object()
                    tpu_object_m.cx = c_x
                    tpu_object_m.cy = c_y
                    tpu_object_m.width = width
                    tpu_object_m.height = height
                    tpu_object_m.label = self.labels[obj.id]
                    tpu_objects_msg.tpu_objects.append(tpu_object_m)

                

  
                    #draw.text((box[0] + (box[2]-box[0]), box[1]), self.labels[obj.label_id] , fill='green')
            
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
        open_cv_image = np.array(prepimg) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        msg.data = np.array(cv2.imencode('.jpg', open_cv_image)[1]).tostring()
        #msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        self.tpu_objects_pub.publish(tpu_objects_msg)

        
        #self.subscriber.unregister()

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
