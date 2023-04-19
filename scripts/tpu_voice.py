#!/usr/bin/env python3
import sys, time, os
import numpy as np
import cv2
import io
import json

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

try:
    from pycoral.adapters import classify
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite

VERBOSE=False

class image_feature:
    def __init__(self, path):
        # topic where we publish
        self.labels = self.load_labels(path + '/labels.txt') 
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
        
        #Init Node
        rospy.init_node('voice_class', anonymous=False)

        # To publish topic
        self.mfcc_pub = rospy.Publisher("/output/image_detected/compressed", CompressedImage, queue_size = 5, tcp_nodelay=False)
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)

        
        # project info
        self.nFrame = goal.duration*FRAME_PER_SEC
        self.projectid = goal.projectid
    
        self.size = 224, 224
        
    def is_silent(self, snd_data, thres):
        frames = np.array(snd_data, dtype=np.int16).astype(np.float32)
        volume_norm = np.linalg.norm(frames/65536.0)*10
        return volume_norm < thres

    def callback(self, msg):
        # message data len = 2205
        self.frame_counter += 1
        print(f"tick : {self.frame_counter}")
        if self.is_silent(msg.data, THRESHOLD) == False:
            self.record_started = True
            self._feedback.status = "START_RECORD"
            self._action_server.publish_feedback(self._feedback)

        if self.record_started and self.frame_counter % 20 == 0:
            self._feedback.status = "RECORDING"
            self._action_server.publish_feedback(self._feedback)
        
        if self.record_started:
            self.snd_data.extend(msg.data)

        if self.frame_counter >= self.nFrame:
            print("Unsubscribe")
            self.audio_sub.unregister()
            self.q.put(1)
    
    def draw_mfcc(self, snd_data, sr, img_width = 224, img_height = 224):
        mfcc_feat = mfcc(snd_data, sr, 
            nfft=2048,
            winfunc=np.hanning
        )
        canvas = (224,224)
        im = Image.new('RGBA', canvas, (255, 255, 255, 255))
        draw = ImageDraw.Draw(im)
        mx = 224 / mfcc_feat.shape[0]
        my = 224 / mfcc_feat.shape[1]
        for x, mfcc_row in enumerate(mfcc_feat):
            for y, mfcc_data in enumerate(mfcc_row):
                mfcc_data = int(mfcc_data)
                if mfcc_data >= 0:
                    draw.rectangle([(x * mx , y * my), (x * mx + mx, y * my + my)], fill = (100, mfcc_data * 10, 100, 255))
                else:
                    draw.rectangle([(x * mx , y * my), (x * mx + mx, y * my + my)], fill = (100, 100, -mfcc_data * 10, 255))
        return im

    def running(self):
        while not rospy.is_shutdown():
            # wait for voice active
            self.frame_counter = 0
            self.record_started = False
            self.snd_data = []
            self.q.queue.clear()
            print(f"start with n frame : {self.nFrame}")
            
            self.audio_sub = rospy.Subscriber("audio_int", kidbright_tpu.msg.int1d, self.callback, queue_size=4)

            timeout = time.time() + TIMEOUT_SEC
            while self.q.empty():
                if time.time() > timeout:
                    self.q.put(0)
                    self.audio_sub.unregister()
                    break
                rospy.sleep(0.1)
    
            record_result = self.q.get()
            if record_result == 1:
                print('Number of frames recorded: ' + str(len(self.snd_data)))

            # create mfcc
            im_mfcc = self.draw_mfcc(self.snd_data, SAMPLE_RATE)
            
            # create waveform
            im_wav = self.draw_wave(self.snd_data)
            
        else:
            pass

    
    def load_labels(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            #parse label from project.json
            project_json = os.path.join(os.path.dirname(filename), "project.json")
            with open(project_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                labels = data["project"]["project"]["modelLabel"]
                print("Project Label : ", ",".join(labels))
                return labels

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
        input_np = self.preprocess(input_np)
        input_np = np.expand_dims(input_np, 0)

        self.interpreter.set_tensor(self.input_details["index"], input_np)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details['index'])
        results = np.squeeze(output_data)
        out = results.argsort()[-1:][::-1]

        #out = classify.get_classes(self.interpreter, top_k=1)

        tpu_objects_msg = tpu_objects()
        
        if len(out) == 1:
            if self.labels:
                target_id = out[0]
                target_score = results[out[0]]
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
    
