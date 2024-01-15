#!/usr/bin/env python

# Extracts frames from rosbags and saves them as pngs

import rosbag
import os
import cv2
from tqdm.auto import tqdm as tqdm
from cv_bridge import CvBridge

rosbag_location = "./"
cv_bridge = CvBridge()


def timestamp_str(ts):
    t = ts.secs + ts.nsecs / float(1e9)
    return "{:.12f}".format(t)


filenames = [
    "bags_2021-08-19-08-25-42.bag",  # S11 side-facing, normal
    "bags_2021-08-19-08-28-43.bag",  # S11 side-facing, normal
    "bags_2021-08-19-09-45-28.bag",  # S11 side-facing, normal
    "bags_2021-08-20-10-19-45.bag",  # S11 side-facing, fast
    "bags_2022-03-28-11-51-26.bag",  # S11 side-facing, normal
    "bags_2022-03-28-12-01-42.bag",  # S11 side-facing, fast
    "bags_2022-03-28-12-03-44.bag",  # S11 side-facing, slow
]

for bagname in tqdm(filenames, position=0):
    bagname = bagname.replace(".bag", "")
    with rosbag.Bag(rosbag_location + "/" + bagname + ".bag", "r") as bag:
        topics = bag.get_type_and_topic_info().topics
        for topic_name, topic_info in topics.items():
            if topic_name == "/dvs/image_raw":
                total_num_frames = topic_info.message_count
                print("Found {} frames in rosbag".format(total_num_frames))

        if not os.path.isdir(rosbag_location + bagname + "/frames"):
            os.makedirs(rosbag_location + bagname + "/frames")

        with tqdm(total=total_num_frames, position=1) as pbar:
            for topic, msg, t in bag.read_messages(topics=["/dvs/image_raw"]):
                if msg.header.stamp.secs < 100:
                    pbar.update(1)
                    continue

                cv_image = cv_bridge.imgmsg_to_cv2(msg, "bgr8")

                if cv_image.shape[0] == 260 and cv_image.shape[1] == 346:
                    cv2.imwrite(rosbag_location + bagname + "/frames/" + timestamp_str(msg.header.stamp) + ".png", cv_image)

                pbar.update(1)
