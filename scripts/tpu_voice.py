#!/usr/bin/env python3
import sys, time, os, io, json, base64
import numpy as np
import cv2
from queue import Queue

import roslib
import rospy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# speech detection
from python_speech_features import mfcc

# Ros Messages
import sensor_msgs.msg
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import kidbright_tpu.msg
from kidbright_tpu.msg import tpu_object
from kidbright_tpu.msg import tpu_objects

try:
    from pycoral.adapters import classify
    from pycoral.adapters import common
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite

VERBOSE=False
SAMPLE_RATE = 44100
FRAME_PER_SEC = 20
TIMEOUT_SEC = 10
MFCC_NUM = 16

class image_feature:
    def __init__(self, path, threshold):
        # topic where we publish
        with open(path + '/project.json') as pjson:
            self.project = json.load(pjson)
        self.labels = self.load_labels(path + '/labels.txt') 
        self.nFrame = self.project["project"]["project"]["options"]["duration"] * FRAME_PER_SEC
        self.threshold = threshold

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
        self.mfcc_pub = rospy.Publisher("/output/mfcc", String, queue_size = 5, tcp_nodelay=False)
        self.tpu_objects_pub = rospy.Publisher("/tpu_objects", tpu_objects, queue_size = 5, tcp_nodelay=False)
        self.status_pub = rospy.Publisher("/voice_class/status", String, queue_size=5,tcp_nodelay = False)
        self.ready_pub = rospy.Publisher("/ready", String, queue_size = 5, tcp_nodelay=False)
        self.hot_loaded = False

        self.q = Queue()    
        self.frame_counter = 0
        self.snd_data = []
        self.size = 224, 224

        self.running()
        
    def is_silent(self, snd_data, thres):
        frames = np.array(snd_data, dtype=np.int16).astype(np.float32)
        volume_norm = np.linalg.norm(frames/65536.0)*10
        return volume_norm < thres

    def load_labels(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            #parse label from project.json
            labels = self.project["project"]["project"]["modelLabel"]
            print("Project Label : ", ",".join(labels))
            return labels

    def preprocess(self, img):
        img = img.astype(np.float32)
        img = img / 255.
        img = img - 0.5
        img = img * 2.
        img = img[:, :, ::-1]
        return img

    def audio_callback(self, msg):
        # message data len = 2205
        if self.is_silent(msg.data, self.threshold) == False and self.record_started == False:
            print("start record")
            self.status_pub.publish('START_RECORD')
            self.record_started = True
            
        if self.record_started:
            self.frame_counter += 1
            print(f"tick : {self.frame_counter}")
            self.snd_data.extend(msg.data)

        if self.frame_counter >= self.nFrame:
            print("end record, unsubscribe")
            self.status_pub.publish('END_RECORD')
            self.audio_sub.unregister()
            self.q.put(1)
    
    def draw_mfcc(self, snd_data, sr, img_width = 224, img_height = 224):
        mfcc_feat = mfcc(np.array(snd_data), sr, nfft=2048, winfunc=np.hanning)
        canvas = (224,224)
        im = Image.new('RGB', canvas, (255, 255, 255))
        draw = ImageDraw.Draw(im)
        mx = 224 / mfcc_feat.shape[0]
        my = 224 / mfcc_feat.shape[1]
        for x, mfcc_row in enumerate(mfcc_feat):
            for y, mfcc_data in enumerate(mfcc_row):
                mfcc_data = int(mfcc_data)
                if mfcc_data >= 0:
                    draw.rectangle([(x * mx , y * my), (x * mx + mx, y * my + my)], fill = (100, mfcc_data * 10, 100))
                else:
                    draw.rectangle([(x * mx , y * my), (x * mx + mx, y * my + my)], fill = (100, 100, -mfcc_data * 10))
        return im

    def classify(self, im):
        image_np = np.array(im) 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        input_np = cv2.resize(image_np.copy(), (self.input_width, self.input_height))
        input_np = self.preprocess(input_np)
        input_np = np.expand_dims(input_np, 0)

        self.interpreter.set_tensor(self.input_details["index"], input_np)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details['index'])
        results = np.squeeze(output_data)
        out = results.argsort()[-1:][::-1]
        return out, results

    def running(self):
        print(f"===========================")
        print(f"project : {self.project['project']['project']['id']}")
        print(f"nframe : {self.nFrame}")
        print(f"threshold : {self.threshold}")
        print(f"===========================")
        while not rospy.is_shutdown():
            # wait for voice active
            self.frame_counter = 0
            self.record_started = False
            self.snd_data = []
            self.q.queue.clear()

            self.audio_sub = rospy.Subscriber("audio_int", kidbright_tpu.msg.int1d, self.audio_callback, queue_size=4)
            self.status_pub.publish('START')

            while self.q.empty():
                rospy.sleep(0.1)
    
            record_result = self.q.get()
            if record_result == 1:
                print('Number of frames recorded: ' + str(len(self.snd_data)))
                
                # create mfcc
                im_mfcc = self.draw_mfcc(self.snd_data, SAMPLE_RATE)
                #classify
                out, results = self.classify(im_mfcc)
                # publish ready status
                if self.hot_loaded == False:
                    self.hot_loaded = True
                    self.ready_pub.publish('READY')
                # pub mfcc 
                with io.BytesIO() as buf_mfccf:
                    im_mfcc.save(buf_mfccf, format="JPEG")
                    buf_mfccf.seek(0)
                    mfcc_str = base64.b64encode(buf_mfccf.read()).decode("ascii")
                self.mfcc_pub.publish(mfcc_str)
                
                if len(out) == 1:
                    tpu_objects_msg = tpu_objects()
                    print(out)
                    print(results)
                    self.status_pub.publish('CLASSIFY')
                    if self.labels:
                        target_id = out[0]
                        target_score = results[out[0]]
                        target_label = self.labels[target_id]
                        # text = f"{target_label} {target_score:0.2f}"
                        # image_np = cv2.putText(image_np, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                        tpu_object_m = tpu_object()
                        tpu_object_m.cx = 0
                        tpu_object_m.cy = 0
                        tpu_object_m.width = 0
                        tpu_object_m.height = 0
                        tpu_object_m.label = target_label
                        tpu_object_m.confident = target_score
                        tpu_objects_msg.tpu_objects.append(tpu_object_m)

                        self.tpu_objects_pub.publish(tpu_objects_msg)

if __name__ == '__main__':
    ic = image_feature(sys.argv[1], float(sys.argv[2]))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    
