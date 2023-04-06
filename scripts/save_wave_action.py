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

sampleRate = 44100 # hertz
import base64
THRESHOLD = 10 # in dB
#CLIENT_PROJECT_DIR= '/home/pi/python'
CLIENT_PROJECT_DIR= '/home/pi/kbclientNew/nectec-client/dist'
FRAME_PER_SEC = 20

from queue import Queue

class saveWave(object):
  # create messages that are used to publish feedback/result
  _feedback = kidbright_tpu.msg.recordFeedback()
  _result   = kidbright_tpu.msg.recordResult()

  def __init__(self):
    
    rospy.init_node('save_wave_file')
    self._action_name = rospy.get_name()
    self._as = actionlib.SimpleActionServer(self._action_name, kidbright_tpu.msg.recordAction, execute_cb=self.execute_cb, auto_start = False)
    self.START_REC = False
    self.fileName = rospy.get_param('~file', "sound.wav")
    # 44100*4/2205
    
    

    self.frame_count = 0
    self.timeoutCounter = 0
    self.sampleRate = sampleRate

    # Set parameters - MFCC
    self.snd_data = []
    self.num_mfcc = 16
    self.len_mfcc = 16
    self.q = Queue()
    self._as.start()

  def is_silent(self, snd_data, thres):   
    #xx = np.frombuffer(snd_data)  
    xx = np.frombuffer(base64.b64decode(snd_data), dtype=np.int16).astype(np.float32)
    #print(sum(np.multiply(xx, xx))/len(snd_data))
    volume_norm = np.linalg.norm(xx/65536.0)*10
    return volume_norm  < thres
    
  def callback(self, msg):
    #print("Get data")  
    self.timeoutCounter = self.timeoutCounter + 1
    if self.is_silent(msg.data, THRESHOLD) == False and self.frame_count == 0:
        self.START_REC = True
    if self.timeoutCounter % 20 == 0  and self.START_REC == False:
        print("PUBLISED")
        self._feedback.status = "StartRec"
        self._as.publish_feedback(self._feedback)

    if self.timeoutCounter == 100 and self.START_REC == False:
        self.timeoutCounter = 0
        print("TERMINATED")
        self._feedback.status = "Timeout"
        self._as.publish_feedback(self._feedback)
        self.a1_sub.unregister()
        self.q.put(2)

    if self.frame_count < self.nFrame and self.START_REC == True:    
        self.frame_count += 1
        self.obj.writeframesraw(base64.b64decode(msg.data))

        # Append msg from publisher to list
        da_o = np.frombuffer(base64.b64decode(msg.data), dtype=np.int16)
        print(da_o)
            
        if(self.is_silent(msg.data, THRESHOLD)):
            print("SILENT")
        else:
            print("START REC")
        self.snd_data.extend(da_o)

            # Print log
        print("here")
        self._feedback.status = "Recording"
        self._as.publish_feedback(self._feedback)
        print(self.nFrame)
        print(self.frame_count)

        # Close wav object
        if self.frame_count == self.nFrame :
            self.obj.close()
            
            print('Wav file saved successfully.')
                  
    elif self.frame_count == self.nFrame: # once recording is done
        self._feedback.status = "MFCC"
        self._as.publish_feedback(self._feedback)
        print('Number of frames recorded: ' + str(len(self.snd_data)))
        mfccs = python_speech_features.base.mfcc(np.array(self.snd_data), 
                                        samplerate=self.sampleRate,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=self.num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)


        mfccs = mfccs.transpose()
        np.set_printoptions(suppress=True)
   
        print('MFCC shape: ' + str(mfccs.shape))
   
        np.savetxt(self.MFCCTextFileName, mfccs, fmt='%f', delimiter=",")

        # Create and save MFCC image
        plt.imshow(mfccs, cmap='inferno', origin='lower')
        plt.savefig(self.MFCCImageFileName)
        print('MFCC saved successfully.')

        # Shutdown node
        print("Unsubscribe down")
        self.a1_sub.unregister()
        self._feedback.status = "Done"
        self._as.publish_feedback(self._feedback)
        self.q.put(1)

 


    
  def execute_cb(self, goal):


    destination = os.path.join(CLIENT_PROJECT_DIR ,goal.projectname ,'audios' )
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    self.frame_count = 0
    self.timeoutCounter = 0

    # Get recording parameters
    
    self.fileName = os.path.join(destination,'wav', dt_string + '.wav')
    self.MFCCTextFileName = os.path.join(destination,'mfcc/text', dt_string+'.csv')
    self.MFCCImageFileName = os.path.join(destination,'mfcc/image', dt_string+'.jpg')
    
    self.nFrame = goal.duration*FRAME_PER_SEC
    print("Goal duration = ")
    print(goal.duration)
    # 44100*4/2205
    
    print(self.nFrame)

    self.frame_count = 0
    
    # append the seeds for the fibonacci sequence
    print(goal)

    self.a1_sub = rospy.Subscriber("a1", String, self.callback, queue_size=4)
    
    # start executing the action

    self.obj = wave.open(self.fileName,'w')
    self.obj.setnchannels(1) # mono
    self.obj.setsampwidth(2)
    self.obj.setframerate(self.sampleRate)
    self.START_REC = False
    self.timeoutCounter = 0
    self.once = True
    self.snd_data = []
    

    self._feedback.status = "Runing"
    # publish the feedback
    self._as.publish_feedback(self._feedback)
    timeout = time.time() + 15
    while self.q.empty():
        print("tick")
        if time.time() > timeout:
            print("timeout")
            break
        rospy.sleep(1)
    rr = self.q.get()
    print("res")
    
    if rr == 1:
        self._result.result = "Done"
    else:
        self._result.result = "TimeOut"
    rospy.loginfo('%s: Succeeded' % self._action_name)
    self._as.set_succeeded(self._result)
      
if __name__ == '__main__':
  saveWave()
  rospy.spin()
