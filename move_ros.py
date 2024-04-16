import rospy
import argparse
from std_msgs.msg import String


def move(message):
    rospy.init_node('dialog_publisher', anonymous=True)
    pub = rospy.Publisher('direction', String, queue_size=10)
    rospy.sleep(0.5)  # Sleep to ensure the publisher is set up

    msg = String()
    msg.data = message
    pub.publish(msg)

    rospy.sleep(1)  # Sleep to ensure the message is sent before shutting down
    rospy.signal_shutdown('Message sent, shutting down dialog node.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send a ROS message.')
    parser.add_argument('message', type=str, help='The message to send')
    args = parser.parse_args()

    move(args.message)