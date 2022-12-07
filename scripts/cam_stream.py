#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

def camThread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    cap.set(cv2.CAP_PROP_FPS, 10)
    image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=5, tcp_nodelay=False)
    rospy.init_node('cam_stream', anonymous=False)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
     
        ret, color_image = cap.read()
        if not ret:
            print("no image")
            continue
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', color_image)[1]).tostring()
        # Publish new image
        image_pub.publish(msg)
        
        #rate.sleep()

    cap.release() 


if __name__ == '__main__':


    try:
        camThread()

    except rospy.ROSInterruptException:
        cam.release() 
        pass
