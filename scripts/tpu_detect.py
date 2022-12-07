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
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

#YOLO
from decoder import YoloDecoder
from box import to_minmax

VERBOSE=False

class image_feature:

    def __init__(self, path):
        '''Initialize ros publisher, ros subscriber'''
        self.labels = read_label_file(path + '/output/labels.txt')
        self.anchors = path + "/output/anchors.txt"
        self.interpreter = make_interpreter(path + '/output/YOLO_best_mAP_edgetpu.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        _, self.input_height, self.input_width, _ = self.input_details['shape']
        self.output_details = self.interpreter.get_output_details()[0]
        
        #YOLO 
        self.decoder = YoloDecoder(self.anchors)
        
        #Publish
        self.image_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)
        #Subscribe
        self.subscriber = rospy.Subscriber("/output/image_raw/compressed", CompressedImage, self.callback,  queue_size = 5, tcp_nodelay=False)

        self.vel_msg = Twist()
        rospy.init_node('image_feature', anonymous=False)

    def ReadAnchorFile(self, file_path):
        return []

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
    
    def bbox_to_xy(self,boxes,w,h):
        #height, width = image.shape[:2]
        minmax_boxes = to_minmax(boxes)
        minmax_boxes[:,0] *= w
        minmax_boxes[:,2] *= w
        minmax_boxes[:,1] *= h
        minmax_boxes[:,3] *= h
        return minmax_boxes.astype(np.int)

    def draw_boxes(self, image, boxes, probs, labels):
        for box, classes in zip(boxes, probs):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(image, 
                        '{}:  {:.2f}'.format(labels[np.argmax(classes)], classes.max()), 
                        (x1, y1 - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,0,255), 1)
        return image        



    def callback(self, ros_data):
        t1 = time.time()
        
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 

        input_np = cv2.resize(image_np.copy(), (self.input_width, self.input_height))
        input_np = np.expand_dims(image_np, 0)

        self.interpreter.set_tensor(self.input_details["index"], input_np)
        self.interpreter.invoke()
        
        output_data = self.interpreter.tensor(self.output_details['index'])().flatten()
        boxes, probs = decoder.run(netout, threshold)
        if len(boxes) > 0:
            boxes = bbox_to_xy(boxes,image.shape[1],image.shape[0])
            if self.labels:
                text = f"{self.labels[obj.id]} {obj.score:0.2f} {c_x:.2f} {area:.2f}"
                    
                tpu_object_m = tpu_object()
                tpu_object_m.cx = c_x
                tpu_object_m.cy = c_y
                tpu_object_m.width = width
                tpu_object_m.height = height
                tpu_object_m.label = self.labels[obj.id]
                tpu_objects_msg.tpu_objects.append(tpu_object_m)
                
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
