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
from datetime import datetime
import time
import base64

SAMPLE_RATE = 44100
THRESHOLD = 10 # in dB
FRAME_PER_SEC = 20
TIMEOUT_SEC = 15
MFCC_NUM = 16

from queue import Queue

class saveWave(object):

  _feedback = kidbright_tpu.msg.recordFeedback()
  _result   = kidbright_tpu.msg.recordResult()

  def __init__(self):
    rospy.init_node('save_wave_action')
    self._action_name = rospy.get_name()
    self._action_server = actionlib.SimpleActionServer(self._action_name, kidbright_tpu.msg.recordAction, execute_cb=self.execute_cb, auto_start = False)
    #---- local var ----
    self.record_started = False
    self.q = Queue()    
    self.frame_counter = 0
    self.timeout_counter = 0
    # Set parameters - MFCC
    self.snd_data = []
    #-------------------
    self._action_server.start()
    
    # 44100*4/2205

#   def is_silent(self, snd_data, thres):   
#     xx = np.frombuffer(base64.b64decode(snd_data), dtype=np.int16).astype(np.float32)
#     volume_norm = np.linalg.norm(xx/65536.0)*10
#     return volume_norm  < thres

#   def callback(self, msg):
#     self.timeoutCounter = self.timeoutCounter + 1
#     if self.is_silent(msg.data, THRESHOLD) == False and self.frame_count == 0:
#         self.START_REC = True
#     if self.timeoutCounter % 20 == 0  and self.START_REC == False:
#         print("PUBLISED")
#         self._feedback.status = "StartRec"
#         self._action_server.publish_feedback(self._feedback)

#     if self.timeoutCounter == 100 and self.START_REC == False:
#         self.timeoutCounter = 0
#         print("TERMINATED")
#         self._feedback.status = "Timeout"
#         self._action_server.publish_feedback(self._feedback)
#         self.a1_sub.unregister()
#         self.q.put(2)

#     if self.frame_count < self.nFrame and self.START_REC == True:    
#         self.frame_count += 1
#         self.obj.writeframesraw(base64.b64decode(msg.data))

#         # Append msg from publisher to list
#         da_o = np.frombuffer(base64.b64decode(msg.data), dtype=np.int16)
#         print(da_o)
            
#         if(self.is_silent(msg.data, THRESHOLD)):
#             print("SILENT")
#         else:
#             print("START REC")
#         self.snd_data.extend(da_o)

#             # Print log
#         print("here")
#         self._feedback.status = "Recording"
#         self._action_server.publish_feedback(self._feedback)
#         print(self.nFrame)
#         print(self.frame_count)

#         # Close wav object
#         if self.frame_count == self.nFrame :
#             self.obj.close()
            
#             print('Wav file saved successfully.')
                  
#     elif self.frame_count == self.nFrame: # once recording is done
#         self._feedback.status = "MFCC"
#         self._action_server.publish_feedback(self._feedback)
#         print('Number of frames recorded: ' + str(len(self.snd_data)))
#         mfccs = python_speech_features.base.mfcc(np.array(self.snd_data), 
#                                         samplerate=self.sampleRate,
#                                         winlen=0.256,
#                                         winstep=0.050,
#                                         numcep=MFCC_NUM,
#                                         nfilt=26,
#                                         nfft=2048,
#                                         preemph=0.0,
#                                         ceplifter=0,
#                                         appendEnergy=False,
#                                         winfunc=np.hanning)


#         mfccs = mfccs.transpose()
#         np.set_printoptions(suppress=True)
   
#         print('MFCC shape: ' + str(mfccs.shape))
   
#         np.savetxt(self.MFCCTextFileName, mfccs, fmt='%f', delimiter=",")

#         # Create and save MFCC image
#         plt.imshow(mfccs, cmap='inferno', origin='lower')
#         plt.savefig(self.MFCCImageFileName)
#         print('MFCC saved successfully.')

#         # Shutdown node
#         print("Unsubscribe down")
#         self.a1_sub.unregister()
#         self._feedback.status = "Done"
#         self._action_server.publish_feedback(self._feedback)
#         self.q.put(1)


  def execute_cb(self, goal):
    self.frame_counter = 0
    self.timeout_counter = 0
    self.record_started = False
    self.snd_data = []

    self.nFrame = goal.duration*FRAME_PER_SEC
    print("Goal = ")
    print(goal)
    print(f"n frame : {self.nFrame}")

    self.audio_sub = rospy.Subscriber("audio_int", kidbright_tpu.msg.int1d, self.callback, queue_size=4)
    
    #----- feedback running ------#
    self._feedback.status = "RUNNING"
    self._action_server.publish_feedback(self._feedback)

    #----- feedback result -------#    
    # timeout = time.time() + TIMEOUT_SEC
    # while self.q.empty():
    #     if time.time() > timeout:
    #         break
    #     rospy.sleep(0.1)
    # rr = self.q.get()
    # if rr == 1:
    #     self._result.result = "Done"
    # else:
    #     self._result.result = "TimeOut"
    
    # rospy.loginfo('%s: Succeeded' % self._action_name)
    # self._action_server.set_succeeded(self._result)
    #------------------------------#  

if __name__ == '__main__':
  saveWave()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down ROS Image feature detector module")
