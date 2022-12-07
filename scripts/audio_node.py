#!/usr/bin/env python3
# license removed for brevity
import rospy

import scipy.signal 

from sys import byteorder
from array import array
from struct import pack
import struct

import pyaudio
import wave
import numpy as np
import actionlib
import kidbright_tpu.msg

#THRESHOLD = 500000
#THRESHOLD = 100000
THRESHOLD = 1000
CHUNK_SIZE = 2205
FORMAT = pyaudio.paInt16
RATE = 44100
import base64


from audio_common_msgs.msg import AudioData
from kidbright_tpu.msg import int1d
from std_msgs.msg import String
from std_msgs.msg import Float32


def is_silent(snd_data, pub, thres):
    "Returns 'True' if below the 'silent' threshold"
    
    #xx = np.frombuffer(snd_data)  
    xx = np.frombuffer(snd_data, dtype=snd_data.typecode).astype(np.float32)
    volume_norm = np.linalg.norm(xx/65536.0)*10
    #rms = np.sqrt(np.mean(np.square(xx)))
    #volume_norm = 20*np.log(rms)/10
    pub.publish(volume_norm)
    #print("Volume in db")
    #print(volume_norm)
    #print(sum(np.multiply(xx, xx))/len(snd_data))
    return sum(np.multiply(xx, xx))/len(snd_data) < thres

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def get_soundcard_dev_info():
   pad_sc = pyaudio.PyAudio()
   max_devs = pad_sc.get_device_count()
   input_devices_index = []
   output_devices_index = []

   for i in range(max_devs):
       devinfo = pad_sc.get_device_info_by_index(i)
       if "TUSBAudio ASIO Driver" in devinfo['name']:
           input_devices_index.append(int(devinfo['index']))
           output_devices_index.append(int(devinfo['index']))

   if not input_devices_index:
      print("NONE")

   print(input_devices_index)
   pad_sc.terminate()

#def action_cb:
    


def talker():
    for x in range(15):
        get_soundcard_dev_info()
    
    #_feedback = kidbright_tpu.msg.recordFeedback()
    #_result = kidbright_tpu.msg.recordResult()

    #pub = rospy.Publisher('audio/audio', AudioData, queue_size=1)
    pub_a = rospy.Publisher('a1', String, queue_size=10)
    pub_aint = rospy.Publisher('audio_int', int1d, queue_size=10)
    pub_sound_db = rospy.Publisher('sound_level', Float32, queue_size=10)

    #_as = actionlib.SimpleActionServer("save_wav", kidbright_tpu.msg.recordAction, execute_cb=action_cb, auto_start = False)
    #_as.start()
    

    rospy.init_node('audio_stream', anonymous=True)
    sampleRate = rospy.get_param('~samplingRate', RATE)
    nchannels = rospy.get_param('~nchannels', 1)
    soundCardNumber = rospy.get_param('~soundCardNumber', 1)
    thres = rospy.get_param('~THRESHOLD', THRESHOLD)
    print(sampleRate)    
    print(nchannels) 
    print(soundCardNumber) 
    print(thres)
    print('----------------------------')
    rate = rospy.Rate(10) # 10hz

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    for i in range (0,numdevices):
        if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

        if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
            print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

    devinfo = p.get_device_info_by_index(1)
    print(devinfo)
    print("Selected device is ",devinfo.get('name'))
    print("Selected device is ",devinfo.get('maxInputChannels'))
    stream = p.open(format=FORMAT, channels=nchannels, rate=sampleRate,
        input=True, output=False, input_device_index=soundCardNumber,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False
    num_chunk = 0
    MAX_CHUNK = 12

    
    while not rospy.is_shutdown():
        snd_data = array('h', stream.read(CHUNK_SIZE))
        #print(len(snd_data))
        
        #snd_data_f = scipy.signal.resample(snd_data_in, 8000 )
        #snd_data = snd_data_f.astype(np.int16)
        #print(type(snd_data_in))
        #print(type(snd_data))
        if byteorder == 'big':
            snd_data.byteswap()

        silent = is_silent(snd_data, pub_sound_db  ,thres)


        if silent and snd_started:
            pass
        elif not silent and not snd_started:
            snd_started = True
        if snd_started:
            #print("snd_started")
        #if True:
            num_chunk += 1
            #print len(snd_data)
            #print len(snd_data)
            #print "num chunk = %d" % num_chunk
            #print silent
            #ad = AudioData()
            #ad.data = tuple(np.fromstring(snd_data, dtype=np.uint8))
            #str1 =  bytes(snd_data) 
            #ad.data = snd_data.tolist()
            #pub.publish(ad)
            #rospy.loginfo(snd_data)
            #bb = np.frombuffer(snd_data, dtype=np.int16).tobytes()
            #print(type(snd_data[0]))
            bb = bytearray(snd_data)
            data_bb = base64.b64encode(bb)
            #print(len(base64.b64encode(bb)))
            cc = int1d()
            #cc.data = np.frombuffer(snd_data, dtype=np.int16)
            cc.data = snd_data
            # print bb
            
            pub_a.publish(data_bb.decode())
            pub_aint.publish(cc)
            
            if num_chunk >= MAX_CHUNK:
                num_chunk = 0
                snd_started = False
        else:
            pass
            

    stream.stop_stream()
    stream.close()
    p.terminate()





if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
