#!/usr/bin/env python3
import rospy
import wave
from std_msgs.msg import String
import numpy as np
import python_speech_features
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
    frames = np.frombuffer(snd_data, dtype=snd_data.typecode).astype(np.float32)
    volume_norm = np.linalg.norm(frames/65536.0)*10
    return volume_norm < thres
  
  def callback(self, msg):
    # message data len = 2205
    self.frame_counter += 1
    
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
      self.audio_sub.unregister()
      self.q.put(1)
      
  def execute_cb(self, goal):
    self.frame_counter = 0
    self.record_started = False
    self.snd_data = []
    print(f".............")
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
        break
      rospy.sleep(0.1)
    
    record_result = self.q.get()
    _result = recordResult()

    if record_result == 1:
      
      print('Number of frames recorded: ' + str(len(self.snd_data)))
      
      # save wav
      with io.BytesIO() as buffer:
        # wavefile = wave.open("__voice.wav", 'w') -- if you want to save this file
        with wave.open(buffer, 'wb') as wav_file:
          wav_file.setnchannels(1) # mono
          wav_file.setsampwidth(2)  # 16-bit audio
          wav_file.setframerate(SAMPLE_RATE)
          wav_file.writeframes(self.snd_data)
        byte_array = bytearray(buffer.getvalue())
        audio_str = base64.b64encode(audio_bytes)
        _result.wav =  audio_str
      
      # # create mfcc
      mfccs = python_speech_features.base.mfcc(np.array(self.snd_data), 
                                    samplerate=self.sampleRate,
                                    winlen=0.256,
                                    winstep=0.050,
                                    numcep=MFCC_NUM,
                                    nfilt=26,
                                    nfft=2048,
                                    preemph=0.0,
                                    ceplifter=0,
                                    appendEnergy=False,
                                    winfunc=np.hanning)
      mfccs = mfccs.transpose()
      plt.imshow(mfccs, cmap='inferno', origin='lower')
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      buf.seek(0)
      mfcc_str = base64.b64encode(buf)
      _result.mfcc = mfcc_str
      
      #plt.savefig(self.MFCCImageFileName)
      
      # with open("img.png", "rb") as image:
      #   f = image.read()
      #   b = bytearray(f)
      #   print b[0]

      _result.result = "SUCCESS"
    else:
      _result.result = "TIMEOUT"

    self._action_server.set_succeeded(_result)

if __name__ == '__main__':
  saveWave()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down ROS Image feature detector module")
