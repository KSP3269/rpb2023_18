#!/usr/bin/env python3
# license removed for brevity
from std_msgs.msg import String
import rospy
import subprocess
import time


def talker():
    time.sleep(0.1)
# TODO: publish 10 string data, topic name is chatter
    pub=rospy.Publisher('chatter',String,queue_size=20)
    rospy.init_node('talker',anonymous=True)
    rate = rospy.Rate(10)
    hello_str="hello world"
    for i in range (10):
        pub.publish(hello_str)
    #rospy.loginfo(hello_str)




    
    
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
