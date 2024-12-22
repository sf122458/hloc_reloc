#! /home/ps/.conda/envs/hloc/bin/python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, String
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

from pathlib import Path
import pycolmap
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.base_model import dynamic_load
from hloc import extractors
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import torch

from gz_db import CAMERA_TOPIC
import time

from utils import quat2rot, rot2quat
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

class Localizer:
    def __init__(self, dataset_name: str):
        print("Localizer starts initializing.")
        # Path definition in hloc
        self.root = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))
        self.images = self.root / f'data/{dataset_name}'
        if not os.path.exists(self.images):
            raise FileExistsError
        db_images = self.images / 'db'
        self.outputs = self.root / f"outputs/{dataset_name}"
        self.sfm_pairs = self.outputs / "pairs-netvlad.txt"
        sfm_dir = self.outputs / "sfm"

        self.feature_conf = extract_features.confs['disk']
        self.matcher_conf = match_features.confs['NN-ratio']
        self.retrieval_conf = extract_features.confs['netvlad']
        self.retrieval_path = self.outputs / 'global-feats-netvlad.h5'

        self.loc_conf = {
            'estimation': {'ransac': {'max_error': 12}},
            'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
        }

        db_image_list = [p.relative_to(self.images).as_posix() for p in (db_images).iterdir()]
        self.model = pycolmap.Reconstruction(sfm_dir)

        Model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retrieval_model = Model(self.retrieval_conf["model"]).eval().to(device)

        self.query = 'query/query.jpg'

        (self.db_desc, self.db_names, self.query_names) = pairs_from_retrieval.prepare(
            self.retrieval_path, db_list=db_image_list, query_list=[self.query])
        

        ######## Cameara ########

        # Since we always use the same camera, the camera params can be determined.
        self.camera = pycolmap.Camera()

        # realsense_d435
        if CAMERA_TOPIC == "/camera/color/image_raw":
            self.camera.width=640
            self.camera.height=480
            self.camera.model=pycolmap.CameraModelId.SIMPLE_RADIAL
            self.camera.params=[768.000000, 320.000000, 240.000000, 0.000000]

        # iris in gazebo
        else:
            self.camera.width=1920
            self.camera.height=1080
            self.camera.model=pycolmap.CameraModelId.SIMPLE_RADIAL
            self.camera.params=[1679.883056, 960.000000, 540.000000, -0.002133]

        ######## Calibration ########

        self.R_point2local = None
        self.T_point2local = None
        self.S = None
        self.q_point2local = None
        self.q_cam2drone = None

        self.calibration = {
            "xyz_cam_in_point": [],
            "xyz_cam_in_local": [],
            "q_cam": [],
            "q_drone": []
        }

        ######## ROS topic ########
        
        rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.pose_cb
        )
        # 当前位置
        self.local_pos = PoseStamped()

        rospy.Subscriber("add_cali",
                        data_class=Empty,
                        callback=self.cali_record_callback)
        
        rospy.Subscriber("calibrate",
                        data_class=Empty,
                        callback=self.cali_callback)
        
        rospy.Subscriber("reloc",
                        data_class=String,      # the path of the image to be relocated
                        callback=self.reloc_callback)


        print("Localizer finishes initializing.")

    def pose_cb(self, msg: PoseStamped):
        self.local_pos = msg

    def hloc_reloc(self):
        time_start = time.time()
        extract_features.extract_feature_from_query(
            self.retrieval_conf,
            self.images,
            query_name=self.query,
            model=self.retrieval_model,
            export_dir=self.outputs
        )
        pairs_from_retrieval.fast_retrieval(
            self.retrieval_path,
            self.sfm_pairs,
            num_matched=5,
            db_desc=self.db_desc,
            db_names=self.db_names,
            query_names=self.query_names
        )
        feature_path = extract_features.main(
            self.feature_conf, 
            self.images,
            self.outputs,
            image_list=[self.query],
            overwrite=True)
        match_path = match_features.main(
            self.matcher_conf,
            self.sfm_pairs,
            self.feature_conf["output"],
            self.outputs,
            overwrite=True
        )

        references_registered = self.get_pairs_info(self.sfm_pairs)
        ref_ids = [self.model.find_image_with_name(name).image_id for name in references_registered]
        localizer = QueryLocalizer(self.model, self.loc_conf)
        ret, log = pose_from_cluster(localizer, self.query, self.camera, ref_ids, feature_path, match_path)
        print(f"Inference time: {time.time() - time_start} s")

        # shape = (1, 3)
        return self.cam2world(ret["cam_from_world"])

    # utils
    def cam2world(self, rigid):
        assert isinstance(rigid, pycolmap.Rigid3d)
        return -np.linalg.inv(quat2rot(rigid.rotation.quat)) @ rigid.translation

    def get_pairs_info(self, file_path: Path):
        pairs_info = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    pairs_info.append(parts[1])
        return pairs_info
    

    def record_query_image(self):
        """
            记录当前图像并保存为query.jpg
        """
        msg = rospy.wait_for_message(CAMERA_TOPIC, Image, timeout=5.0)
        image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        name = os.path.join(self.images, "query/query.jpg")
        cv2.imwrite(name, image)

    def get_transform(self):
        """
            通过以获得的校准图像计算坐标系的变换矩阵
        """
        xyz_a = np.array(self.calibration["xyz_cam_in_point"]).T
        xyz_b = np.array(self.calibration["xyz_cam_in_local"]).T
        # scale = np.linalg.norm(xyz_b) / np.linalg.norm(xyz_a)
        scale = np.sqrt(np.sum(np.square(xyz_b[:, 0] - xyz_b[:, 1])) / np.sum(np.square(xyz_a[:, 0] - xyz_a[:, 1])))
        xyz_a = xyz_a * scale
        self.S = np.diag([scale, scale, scale])
        centroid_a = np.mean(xyz_a, axis=-1, keepdims=True)
        centroid_b = np.mean(xyz_b, axis=-1, keepdims=True)
        H = (xyz_a - centroid_a) @ (xyz_b - centroid_b).T
        U, _, V = np.linalg.svd(H)
        R = V.T @ U.T
        # if np.linalg.det(R) < 0:
        #   V[2, :] *= -1
        #   R = V.T @ U.T
        T = -R @ centroid_a + centroid_b
        self.R_point2local = R
        self.T_point2local = T
        print(f"Scale: {scale}\nRotation: {R}\nTranslation: {T}")
        # self.q_point2local = rot2quat(R)

    def reloc_callback(self, msg: String):
        """
            对已有的图像进行重定位，最终获得目标图像在当前本地坐标系下的坐标
            msg: the path of the image to be relocated
        """
        path = msg.data
        print("Start relocating...")
        if os.path.exists(path):
            os.system(f"cp {path} {self.images}/query/query.jpg")
        xyz_cam_in_point = self.hloc_reloc()
        # TODO: transform the camera pose in the point-cloud coordinate to the world coordinate
        xyz_cam_in_world = self.R_point2local @ self.S @ np.array([xyz_cam_in_point]).T + self.T_point2local
        print(f"Camera pose in the world coordinate: {xyz_cam_in_world}")

    def cali_record_callback(self, msg: Empty):
        # 记录校准图像
        print("Start recording...")
        self.record_query_image()
        coord_in_local = np.array([self.local_pos.pose.position.x,
                        self.local_pos.pose.position.y,
                        self.local_pos.pose.position.z])
        coord_in_point = self.hloc_reloc()
        
        self.calibration["xyz_cam_in_point"].append(coord_in_point)

        # TODO: record the camera pose in the local coordinate
        self.calibration["xyz_cam_in_local"].append(coord_in_local)
        
        print("Coord in point-cloud coordinate: ", coord_in_point
              , "\nCoord in local coordinate: ", coord_in_local)
    

    def cali_callback(self, msg: Empty):
        # 开始校准
        print("Start calibrating...")
        self.get_transform()
        # TODO: calculate the quaternion transform


if __name__ == "__main__":
    
    rospy.init_node("camera_test")
    rate = rospy.Rate(20)
    # FIXME: dataset name
    localizer = Localizer(dataset_name="town")
    while not rospy.is_shutdown():
        rate.sleep()