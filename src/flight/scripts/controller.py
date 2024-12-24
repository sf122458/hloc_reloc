#! /home/ps/.conda/envs/hloc/bin/python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import String, Float32
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import dist, unwrap_pose

class BasicDroneController:
    def __init__(self, rate=20):

        self.rate = rospy.Rate(rate)

        self.current_state = State()
        self.local_pos = PoseStamped()
        self.target_pos = PoseStamped()
        self.local_vel = TwistStamped()

        # 订阅无人机位置与姿态xyz, q
        rospy.Subscriber("mavros/local_position/pose", PoseStamped, self.local_pos_cb)

        # 发布无人机位置与姿态xyz, q
        self.local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)


        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb)

        rospy.wait_for_service("mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

        rospy.wait_for_service("mavros/set_mode")
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        rospy.Subscriber("set_cmd", String, self.cmd_cb)
        rospy.Subscriber("set_pos", PoseStamped, self.set_pos_cb)
        rospy.Subscriber("set_pos_delta", PoseStamped, self.set_pos_delta_cb)
        rospy.Subscriber("set_yaw", Float32, self.set_yaw_cb)

    # callback functions
    def state_cb(self, msg):
        self.current_state = msg

    def local_pos_cb(self, msg):
        self.local_pos = msg

    def cmd_cb(self, msg: String):
        if msg.data == "takeoff":
            self.takeoff()
        elif msg.data == "land":
            self.land()

    def set_pos_cb(self, msg: PoseStamped):
        self.target_pos.pose.position.x = msg.pose.position.x
        self.target_pos.pose.position.y = msg.pose.position.y
        self.target_pos.pose.position.z = msg.pose.position.z

    def set_pos_delta_cb(self, msg: PoseStamped):
        if abs(msg.pose.position.x) > 3 or abs(msg.pose.position.y) > 3 or abs(msg.pose.position.z) > 3:
            rospy.logwarn("Delta position too large, set_pos_delta ignored")
            return
        if self.target_pos.pose.position.z + msg.pose.position.z < 0:
            rospy.logwarn("Target position below ground, set_pos_delta ignored")
            return
        self.target_pos.pose.position.x += msg.pose.position.x
        self.target_pos.pose.position.y += msg.pose.position.y
        self.target_pos.pose.position.z += msg.pose.position.z

    def set_yaw_cb(self, msg: Float32):
        x, y, z, w = R.from_euler('ZYX', [msg.data/180*np.pi, 0, 0]).as_quat()
        self.target_pos.pose.orientation.x = x
        self.target_pos.pose.orientation.y = y
        self.target_pos.pose.orientation.z = z
        self.target_pos.pose.orientation.w = w

    # basic cmd
    def init_stream(self, rate, n=100):
        for _ in range(n):
            if(rospy.is_shutdown()):
                break
            pose = PoseStamped()
            self.local_pos_pub.publish(pose)
            rate.sleep()

    def arm(self):
        while not self.current_state.armed:
            self.arming_client(True)
        print("Vehicle Armed")

    def land(self):
        rospy.loginfo("Landing")
        self.set_mode("AUTO.LAND")

    def set_mode(self, mode):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = mode
        result = self.set_mode_client.call(offb_set_mode)
        if result.mode_sent:
            rospy.loginfo("Setting mode to %s successful", mode)
        else:
            rospy.loginfo("Setting mode to %s unsuccessful", mode)

    def takeoff(self, z=1):
        rospy.loginfo("Connecting to Autopilot")
        while not self.current_state.connected:
            self.rate.sleep()

        self.arm()

        # NOTE: enter OFFBOARD mode
        self.init_stream(self.rate)
        
        rospy.loginfo("Taking off")

        self.target_pos = PoseStamped()
        self.target_pos.pose.position.z = z
        # x, y, z, w = R.from_euler('ZYX', [np.pi/2*0.8, 0, 0]).as_quat()
        # self.target_pos.pose.orientation.x = x
        # self.target_pos.pose.orientation.y = y
        # self.target_pos.pose.orientation.z = z
        # self.target_pos.pose.orientation.w = w

        while dist(unwrap_pose(self.local_pos)[0], unwrap_pose(self.target_pos)[0]) > 0.1:
            if self.current_state.mode != "OFFBOARD":
                self.set_mode("OFFBOARD")

            self.local_pos_pub.publish(self.target_pos)
            self.rate.sleep()


    def main(self):
        last_req = rospy.Time.now()
        while not rospy.is_shutdown():

            if rospy.Time.now() - last_req > rospy.Duration(5.0):
                last_req = rospy.Time.now()
                # print(f"Position: x:{self.local_pos.pose.position.x}, y:{self.local_pos.pose.position.y}, z:{self.local_pos.pose.position.z}")
                # print(f"Orientation: x:{self.local_pos.pose.orientation.x}, y:{self.local_pos.pose.orientation.y}, z:{self.local_pos.pose.orientation.z}, w:{self.local_pos.pose.orientation.w}")
                
            self.local_pos_pub.publish(self.target_pos)
            self.rate.sleep()


            

if __name__ == "__main__":
    rospy.init_node("mavros_drone_test")

    drone = BasicDroneController()

    # drone.takeoff(1)
    drone.main()

    rospy.spin()