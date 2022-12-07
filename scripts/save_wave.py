#!/usr/bin/env python3
import rospy
import wave
from std_msgs.msg import String
import numpy as np
import python_speech_features
import matplotlib
matplotlib.use('Agg') # to suppress plt.show()
import matplotlib.pyplot as plt

sampleRate = 44100 # hertz
import base64
THRESHOLD = 10 # in dB

class save_wave():
    def __init__(self):
      
        rospy.init_node("wave_wait", anonymous=False)
        self.pub_status = rospy.Publisher('record_status', String, queue_size=1)
        #rospy.wait_for_message("audio/audio", AudioData)

        # Set parameters - wav
        self.sampleRate = rospy.get_param('~samplingRate', sampleRate)
        self.fileName = rospy.get_param('~file', "sound.wav")
        # 44100*4/2205
        self.nFrame = rospy.get_param('~nframe', 20)
        print(self.fileName)

        self.frame_count = 0

        # Set parameters - MFCC
        self.snd_data = []
        self.num_mfcc = 16
        self.len_mfcc = 16
        self.MFCCTextFileName = rospy.get_param('~mfcc_text_file', "foo_2.csv")
        self.MFCCImageFileName = rospy.get_param('~mfcc_image_file', "mfcc.jpg")

        #self.number_subscriber = rospy.Subscriber("audio/audio", AudioData, self.callback, queue_size=1)
        # Subscribe to "a1" publisher
        self.a1_sub = rospy.Subscriber("a1", String, self.callback, queue_size=4)
        
        # Set wav object
        rospy.loginfo("Record wave file")
        self.obj = wave.open(self.fileName,'w')
        self.obj.setnchannels(1) # mono
        self.obj.setsampwidth(2)
        self.obj.setframerate(self.sampleRate)
        self.START_REC = False
        self.timeoutCounter = 0
        self.once = True
        rate = rospy.Rate(1)
        
        
        while not rospy.is_shutdown():
            rate.sleep()



        # Set MFCC object
        # self.mfcc_obj = np.savefig???
        

    def is_silent(self, snd_data, thres):
        "Returns 'True' if below the 'silent' threshold"
    
        #xx = np.frombuffer(snd_data)  
        xx = np.frombuffer(base64.b64decode(snd_data), dtype=np.int16).astype(np.float32)
        #print(sum(np.multiply(xx, xx))/len(snd_data))
        volume_norm = np.linalg.norm(xx/65536.0)*10
        return volume_norm  < thres

    def callback(self, msg):

        self.timeoutCounter = self.timeoutCounter + 1
        if self.is_silent(msg.data, THRESHOLD) == False and self.frame_count == 0:
            self.START_REC = True
        if self.timeoutCounter % 20 == 0  and self.START_REC == False:
            print("PUBLISED")
            self.pub_status.publish("start_record")
        if self.timeoutCounter == 100 and self.START_REC == False:
            self.timeoutCounter = 0
            print("TERMINATED")
            self.pub_status.publish("node_terminated")
            rospy.signal_shutdown("Term")

        #print("-----------------------------------------")
        #print(self.START_REC)
        #print("=========================================")
        #print(self.frame_count < self.nFrame and self.START_REC == True)
        if self.frame_count < self.nFrame and self.START_REC == True:    
            
            # Write msg from publisher to wav
            self.frame_count += 1
            #print("***************************************")
            #print(len(msg.data))
            #print(len(base64.b64decode(msg.data)))

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
            print(self.nFrame)
            print(self.frame_count)

            # Close wav object
            if self.frame_count == self.nFrame :
                self.obj.close()
            
                print('Wav file saved successfully.')
                  
        elif self.frame_count == self.nFrame: # once recording is done
            #self.frame_count = 0

            # Convert snd_data to MFCC and save it
            # print type(self.snd_data[0])
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

            # Transpose MFCC, so that it is a time domain graph
            #print("min val = " + str(mfccs_1.min()))
            #print("mac val = " + str(mfccs_1.max()))
            #mfccs_1 -= mfccs_1.min()
            #mfccs_1 += 0.0001
            #mfccs_1 /= mfccs_1.max() 
            #mfccs = np.log(mfccs_1)
            mfccs = mfccs.transpose()
            np.set_printoptions(suppress=True)
            # print type(mfccs)
            print('MFCC shape: ' + str(mfccs.shape))
            #print(mfccs)
            # print np.matrix(mfccs)
            #np.savetxt('array_hf.csv', [mfccs], delimiter=',' , header='A Sample 2D Numpy Array :: Header', footer='This is footer')
            # np.savetxt("foo.csv", mfccs, fmt='%f', delimiter=",")

            # Save MFCC text file - used for training
            np.savetxt(self.MFCCTextFileName, mfccs, fmt='%f', delimiter=",")

            # Create and save MFCC image
            plt.imshow(mfccs, cmap='inferno', origin='lower')
            plt.savefig(self.MFCCImageFileName)
            print('MFCC saved successfully.')

            # Shutdown node
            print("Shuttting down")
            self.pub_status.publish("done_record")
            rospy.signal_shutdown("Term")
    

if __name__ == '__main__':
    print("hello")
    try:
        save_wave()
        
        #rospy.spin()
    except:
        print("except")
        pass




