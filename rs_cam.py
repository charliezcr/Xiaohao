import pyrealsense2 as rs
import numpy as np
import os
import cv2
import mmap
import rospy
from std_msgs.msg import String, Int8

shm_fd = os.open('/dev/shm/camera_6', os.O_CREAT | os.O_TRUNC | os.O_RDWR)
shm_fd2 = os.open('/dev/shm/camera_7', os.O_CREAT | os.O_TRUNC | os.O_RDWR)


def callback(data):
    global align, pub_direction, rate
    # locate item and move towards it
    if data.data.startswith('locate'):
        # get the 2d coordinates
        coord = data.data.split(':')[1:3]
        x = int(coord[0])
        y = int(coord[1])
        aligned_frames = align.process(frames)  # get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # get depth frame
        dis = aligned_depth_frame.get_distance(x, y)
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # get 3D coordinates
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)
        print(camera_coordinate)
        # check whether turn left or right
        direction = 'right_'
        if camera_coordinate[0] < 0:
            direction = 'left_'
        # calculate the angle to turn
        angle_offset_x = np.arctan2(camera_coordinate[0], camera_coordinate[2])
        angle_offset_x_deg = int(np.abs(np.degrees(angle_offset_x)))
        turn_msg = direction + str(angle_offset_x_deg) + '_degree'
        print(turn_msg)
        pub_direction.publish(turn_msg)
        rate.sleep()
        rate.sleep()
        # calculate the distance to move
        hypotenuse = np.sqrt(camera_coordinate[0] ** 2 + camera_coordinate[2] ** 2)
        distance = round(hypotenuse, 2)
        if distance > 1.2:
            distance -= 0.4
        elif distance > 0.5:
            distance -= 0.3
        else:
            distance *= 0.8
        # move
        march_msg = 'forward_' + str(distance) + '_m'
        print(march_msg)
        pub_direction.publish(march_msg)
        rate.sleep()


if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    # Start streaming
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    rospy.init_node('realsense_camera_subscriber', anonymous=True)
    pub_direction = rospy.Publisher('/direction', String, queue_size=10)
    rate = rospy.Rate(1)
    rospy.Subscriber("camera", String, callback)

    try:
        width = int(640)
        height = int(480)
        shm_size = width * height * 2
        shm_size2 = width * height * 3
        os.ftruncate(shm_fd, shm_size)
        os.ftruncate(shm_fd2, shm_size2)
        mmap_data = mmap.mmap(shm_fd, shm_size)
        mmap_data2 = mmap.mmap(shm_fd2, shm_size2)
        print("mmap")
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))
            images = cv2.flip(images, -1)
            mmap_data.seek(0)
            mmap_data.write(depth_image.astype(np.uint16).tobytes())
            mmap_data2.seek(0)
            mmap_data2.write(color_image.tobytes())
        rospy.spin()
    finally:
        # Stop streaming
        pipeline.stop()
