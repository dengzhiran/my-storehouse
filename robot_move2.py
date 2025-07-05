# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import rospy
import socket
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# 初始化 ROS 节点
rospy.init_node('robot_motion_control', anonymous=True)

# 发布机器人运动命令
cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# 发布跟踪状态信息（给上位机或其他节点）
track_pub = rospy.Publisher('/track_command', String, queue_size=10)

# 是否正在跟踪
tracking_active = False


def control_by_box(center_x, box_width, image_width):
    twist = Twist()

    # 控制参数
    Kp_linear = 0.1
    Kp_angular = 0.002  # 小一点，防止转太快
    min_box_size = 100
    max_box_size = 300
    angular_dead_zone = 30  # 死区

    # 控制线速度
    if box_width < min_box_size:
        twist.linear.x = 0.2
    elif box_width > max_box_size:
        twist.linear.x = 0.0
    else:
        twist.linear.x = Kp_linear * (box_width - min_box_size)

    # 控制角速度（根据偏移量 * Kp）
    offset = center_x - image_width / 2
    if abs(offset) > angular_dead_zone:
        twist.angular.z = -Kp_angular * offset  # 注意符号：左偏为正 -> 左转（负角速度）
    else:
        twist.angular.z = 0.0

    rospy.loginfo(u"控制命令 - linear.x: {:.2f}, angular.z: {:.2f}".format(twist.linear.x, twist.angular.z))
    cmd_pub.publish(twist)

# 统一处理所有来自上位机的控制信息
def process_control_command(msg):
    global tracking_active
    msg = msg.strip()

    if msg == "hand_raised":
        tracking_active = True
        rospy.loginfo("收到 hand_raised，开始跟踪")
        track_pub.publish("tracking started")
        # 可以让机器人先动一下表示状态切换
        twist = Twist()
        twist.linear.x = 0.2
        cmd_pub.publish(twist)

    elif msg == "clap":
        tracking_active = False
        rospy.loginfo("收到 clap，停止跟踪")
        track_pub.publish("tracking stopped")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        cmd_pub.publish(twist)

    else:
        if tracking_active:
            try:
                center_x, box_width = list(map(int, msg.split(",")))  # 转换为列表
                image_width = 640  # 设定图像宽度
                control_by_box(center_x, box_width, image_width)
            except Exception as e:
                rospy.logwarn("无法解析识别框信息: '%s'，错误: %s" % (msg, e))
        else:
            rospy.loginfo("未处于跟踪状态，忽略识别框信息: %s" % msg)

# 启动 socket 服务端并监听上位机信息
HOST = '0.0.0.0'
PORT = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("服务器启动，等待连接...（监听端口 %d)" % PORT)
conn, addr = s.accept()
print("连接来自：", addr)

while not rospy.is_shutdown():
    data = conn.recv(1024)
    if not data:
        break
    msg = data.decode('utf-8')
    print("收到消息：%s" % msg)
    process_control_command(msg)
    response = "虚拟机收到：%s" % msg
    conn.sendall(response.encode('utf-8'))

conn.close()
s.close()
