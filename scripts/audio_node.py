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
import base64

from audio_common_msgs.msg import AudioData
from kidbright_tpu.msg import int1d
from std_msgs.msg import String
from std_msgs.msg import Float32

#=========== AUDIO CONFIG ==========#
THRESHOLD = 12 #in db 0 - 140 db
CHUNK_SIZE = 2205
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
N_CHANNEL = 1
MAX_CHUNK = 50
#===================================#


def is_silent(snd_data, pub, thres):
    xx = np.frombuffer(snd_data, dtype=snd_data.typecode).astype(np.float32)
    volume_norm = np.linalg.norm(xx/65536.0)*10
    pub.publish(volume_norm)
    return volume_norm < thres

def find_device():
    p = pyaudio.PyAudio()
    max_devs = p.get_device_count()
    device_index = -1
    for i in range(max_devs):
        devinfo = p.get_device_info_by_index(i)
        if "USB Audio Device" in devinfo['name']:
            device_index = int(devinfo['index'])
            break
    return device_index, devinfo

def audio_node():
    pub_a = rospy.Publisher('a1', String, queue_size=10)
    pub_aint = rospy.Publisher('audio_int', int1d, queue_size=10)
    pub_sound_db = rospy.Publisher('sound_level', Float32, queue_size=10)

    rospy.init_node('audio_stream', anonymous=False) # singleton only audio stream can publish audio

    device_index = -1
    max_retry = 5
    devinfo = None

    while device_index < 0 and max_retry > 0:
        print(f"========= FIND DEVICE ({max_retry}) =======")
        device_index, devinfo = find_device()
        max_retry = max_retry - 1
        rospy.sleep(1)

    if device_index < 0:
        print("====== NO USB RECORD DEVICE !!! - TERMINATE ======")
        return

    print("======= start audio node with params ========")
    print(f"Card Device Index : {device_index}") 
    print(f"Sample Rate : {SAMPLE_RATE}")    
    print(f"Channel : {N_CHANNEL}") 
    print(f"Threshold : {THRESHOLD}")
    print(f"Chunk Size : {CHUNK_SIZE}")
    print(f"Selected Device Name : {devinfo.get('name')}")
    print(f"Selected Device Max CH : {devinfo.get('maxInputChannels')}")
    print('=============================================')
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=N_CHANNEL, rate=SAMPLE_RATE,
        input=True, output=False, input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE)

    record_started = False
    num_chunk = 0

    while not rospy.is_shutdown():
        snd_data = array('h', stream.read(CHUNK_SIZE)) #len snd_data = 2205

        if byteorder == 'big':
            snd_data.byteswap()

        silent = is_silent(snd_data, pub_sound_db  ,THRESHOLD)

        if not record_started and not silent: # when voice and ready state
            record_started = True
            print("start record")

        if record_started:
            num_chunk += 1
            #--- base64 publish
            audio_bytes = bytearray(snd_data)
            audio_str = base64.b64encode(audio_bytes)
            pub_a.publish(audio_str)

            #--- integer publish
            audio_int1d = int1d()
            audio_int1d.data = snd_data
            pub_aint.publish(audio_int1d)

            #--- stop record
            if num_chunk >= MAX_CHUNK:
                num_chunk = 0
                record_started = False
                print("end record")

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    try:
        audio_node()
    except rospy.ROSInterruptException:
        pass
