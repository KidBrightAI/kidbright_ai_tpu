#!/usr/bin/env python3
import rospy
import wave
from std_msgs.msg import String
import numpy as np
from python_speech_features import mfcc
import matplotlib
matplotlib.use('Agg') # to suppress plt.show()
import matplotlib.pyplot as plt
import actionlib
from kidbright_tpu.msg import recordAction
from kidbright_tpu.msg import recordGoal
from kidbright_tpu.msg import recordResult
from kidbright_tpu.msg import recordFeedback
import kidbright_tpu.msg
import os
import io
from datetime import datetime
import time
import base64
import struct

from PIL import Image, ImageDraw

SAMPLE_RATE = 44100
THRESHOLD = 10 # in dB
FRAME_PER_SEC = 20
TIMEOUT_SEC = 10
MFCC_NUM = 16

from queue import Queue

class saveWave(object):
  
  def __init__(self):
    rospy.init_node('save_wave_action')
    self._action_name = rospy.get_name()
    self._action_server = actionlib.SimpleActionServer(self._action_name, kidbright_tpu.msg.recordAction, execute_cb=self.execute_cb, auto_start = False)
    #---- local var ----
    self.record_started = False
    self.q = Queue()    
    self.frame_counter = 0
    self.snd_data = []
    #-------------------
    self._feedback = recordFeedback()
    print("start record action server")
    self._action_server.start()    
    # 44100*4/2205

  def is_silent(self, snd_data, thres):
    frames = np.array(snd_data, dtype=np.int16).astype(np.float32)
    volume_norm = np.linalg.norm(frames/65536.0)*10
    return volume_norm < thres

  def draw_wave(self, snd_data, img_width = 700, img_height = 100):
    line_color = 180
    rms_len = 128
    section_len = len(snd_data) // rms_len
    mx = img_width / rms_len
    im = Image.new('RGB',(img_width, img_height),color="black")
    draw = ImageDraw.Draw(im)
    for x in range(rms_len - 1):
        st = x * section_len
        sp = st + section_len
        block_rms = np.sqrt(np.mean(snd_data[st:sp]**2))
        v = int(block_rms)/100 + 2
        y1 = (img_height - v) // 2
        y2 = y1 + v
        draw.rectangle([(x * mx , y1), (x * mx + mx, y2)], fill = "#FFA500",outline="#222")
    return im

  def draw_mfcc(self, snd_data, sr, img_width = 224, img_height = 224):
    mfcc_feat = mfcc(snd_data, sr, nfft=2048, winfunc=np.hanning)
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
      
  def execute_cb(self, goal):
    print("------------")
    self.frame_counter = 0
    self.record_started = False
    self.snd_data = []
    self.q.queue.clear()
    
    print(goal)
    self.nFrame = goal.duration*FRAME_PER_SEC
    self.projectid = goal.projectid
    print(f"n frame : {self.nFrame}")

    self.audio_sub = rospy.Subscriber("audio_int", kidbright_tpu.msg.int1d, self.callback, queue_size=4)
    
    #----- feedback running ------#
    self._feedback.status = "RUNNING"
    self._action_server.publish_feedback(self._feedback)

    #----- feedback result -------#    
    timeout = time.time() + TIMEOUT_SEC
    while self.q.empty():
      if time.time() > timeout:
        self.q.put(0)
        self.audio_sub.unregister()
        break
      rospy.sleep(0.1)
    
    record_result = self.q.get()
    _result = recordResult()
    all_result = []
    
    if record_result == 1:
      
      all_result.append("SUCCESS")
      print('Number of frames recorded: ' + str(len(self.snd_data)))

      # save wav
      with io.BytesIO() as buffer:
        with wave.open(buffer, 'wb') as wav_file:
          wav_file.setnchannels(1) # mono
          wav_file.setsampwidth(2)  # 16-bit audio
          wav_file.setframerate(SAMPLE_RATE)
          byte_array = bytearray(struct.pack('h' * len(self.snd_data), *self.snd_data))
          wav_file.writeframes(byte_array)
        
        byte_array = bytearray(buffer.getvalue())
        audio_str = base64.b64encode(byte_array).decode('ascii')
      all_result.append(audio_str)

      # create mfcc
      im_mfcc = self.draw_mfcc(self.snd_data, SAMPLE_RATE)
      with io.BytesIO() as buf_mfccf:
        im_mfcc.save(buf_mfccf, format="JPG")
        buf_mfccf.seek(0)
        mfcc_str = base64.b64encode(buf_mfccf.read()).decode("ascii")
      all_result.append(mfcc_str)

      # create waveform
      im_wav = self.draw_wave(self.snd_data)
      with io.BytesIO() as buf_wavef:
        im_wav.save(buf_wavef, format="JPG")
        buf_wavef.seek(0)
        waveform_str = base64.b64encode(buf_wavef.read()).decode('ascii')
      all_result.append(waveform_str)
      
    else:
      all_result.append("TIMEOUT")
      
    _result.result = "$".join(all_result)
    self._action_server.set_succeeded(_result)

if __name__ == '__main__':
  saveWave()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down ROS Image feature detector module")
