#! /home/ps/.conda/envs/hloc/bin/python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import os

from pathlib import Path

import datetime
import json
"""
- drone_camera (root)
    - data
        - db
            - *.jpg
        - query
            - *.jpg
    - scripts
        - *.py
    - outputs
        - *.h5
        - *.txt

"""

# CAMERA_TOPIC = "/camera/color/image_raw"
CAMERA_TOPIC = "/iris/usb_cam/image_raw"

# 记录photo的位置信息
class LogWriter:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'w') as f:
            json.dump({}, f)
    
    def write(self, new_dict):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.update(new_dict)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

class Localizer:
    def __init__(self, dataset_name: str):
        print("Start initializing.")
        # Path definition
        self.root = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
        self.db_images = self.root / f'data/{dataset_name}_{datetime.datetime.now().strftime("%m-%d-%H:%M:%S")}/db'
        os.makedirs(self.db_images, exist_ok=True)
        
        self.idx = 0
        
        rospy.Subscriber("take_photo", 
                        data_class=Bool,
                        callback=self.callback)
        
        rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.pose_cb
        )

        # 当前位置
        self.local_pos = PoseStamped()

        self.writer = LogWriter(self.db_images / "log.json")

        print("Finish initializing.")

    def pose_cb(self, msg):
        self.local_pos = msg

    def callback(self, msg):
        msg = rospy.wait_for_message(CAMERA_TOPIC, Image, timeout=5.0)
        try:
            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(e)
        name = os.path.join(self.db_images, f"img{self.idx}.jpg")
        cv2.imwrite(name, image)
        info_dict = {
            f"idx{self.idx}.jpg":{
                "px": self.local_pos.pose.position.x,
                "py": self.local_pos.pose.position.y,
                "pz": self.local_pos.pose.position.z,
                'qx': self.local_pos.pose.orientation.x,
                'qy': self.local_pos.pose.orientation.y,
                'qz': self.local_pos.pose.orientation.z,
                'qw': self.local_pos.pose.orientation.w,
            }
        }
        self.writer.write(info_dict)
        print(f"Saving image to {name}.")
        self.idx = self.idx + 1


if __name__ == "__main__":
    
    rospy.init_node("camera")
    rate = rospy.Rate(20)
    localizer = Localizer("test")
    while not rospy.is_shutdown():
        rate.sleep()