#!/usr/bin/env python


import rospy
rospy.init_node('get_center', anonymous=False)

import rosnode
import subprocess
import time
import os
ros_nodes = rosnode.get_node_names()
if not '/image_feature' in ros_nodes:
    command='rosrun kidbright_tpu tpu_detect.py'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    time.sleep(10)



while not rospy.is_shutdown():
    import roslib
    import rospy

    # Ros Messages
    from kidbright_tpu.msg import tpu_object
    from kidbright_tpu.msg import tpu_objects

    objs = rospy.wait_for_message('/tpu_objects', tpu_objects, timeout=4)
    print(objs.label)