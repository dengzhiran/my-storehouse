# my-storehouse
基于单目人体姿态识别机器人特定人员跟踪的实现
mypose2.py         上位机代码，包含人脸识别与姿态识别
robot_move2.py  下位机代码，控制机器人移动，可用虚拟机实体机器人运行       
register_faces.py  人脸存储代码，将人脸图片存入faces文件夹后运行
本项目基于 Python + OpenCV + YOLOv8‑Pose 实现了一个集人脸识别、姿态识别与机器人跟踪控制于一体的视觉感知系统。采用上位机进行图像分析与动作判断，通过 Socket 通信将控制指令传递至下位机，从而实现对特定人员的动作响应式跟踪。

## 📌 项目特点

- ✅ 基于人脸识别启动跟踪，仅响应已注册人员
- ✅ 支持识别抬手启动、击掌停止等动作控制
- ✅ 上下位机基于 Socket 实现实时通信
- ✅ 下位机接收坐标指令并控制机器人运动

## 🧱 系统架构

本系统分为四个主要模块：

1. **感知层（上位机）**：图像识别模块（MTCNN + InceptionResnetV1 + YOLOv8-Pose）
2. **决策层**：逻辑判断与控制指令封装
3. **通信层**：基于 TCP 的 Socket 通信
4. **执行层（下位机）**：根据控制指令驱动底盘移动

## 🚀 快速开始

1.安装依赖

建议使用 Python 3.8+ 环境：

```bash
pip install -r requirements.txt

2.启动下位机程序 
python robot_move2.py

3.启动上位机程序
python mypose2.py
请确保上下位机在 同一局域网 下


通信方式：基于 TCP 的 Socket
上位机发送内容：
人物中心坐标 (x, y)
识别框宽度
当前识别状态（抬手 / 击掌 / 无动作）
下位机根据这些数据控制机器人转向与前进速度

上位机推荐运行在 Windows + 摄像头环境
下位机建议使用 ROS 支持的 Linux 环境（如 Ubuntu）
requirements.txt 未包含 YOLOv8 和 facenet 模型的权重文件
摄像头帧率与处理帧速影响整体响应性能

