import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from ultralytics import YOLO
import socket
import time

# 初始化 TCP 客户端，连接虚拟机（下位机）
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("", 12345))  # 下位机 IP

def send_command(cmd):
    try:
        client_socket.sendall(cmd.encode())
        print(f"已发送命令: {cmd}")
    except Exception as e:
        print("发送失败:", e)

# 初始化人脸模型
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 读取已存储的人脸特征
try:
    saved_face_embeddings = torch.load("saved_faces.pth")
    print(f"成功加载 {len(saved_face_embeddings)} 张已录入的人脸数据")
except FileNotFoundError:
    print("错误：未找到 'saved_faces.pth'，请先运行 'register_faces.py' 录入人脸数据！")
    saved_face_embeddings = {}

# 允许进入姿态识别的人脸列表（文件名或自定义名称）
authorized_face_names = [""] #此处放入图片名称

# 人脸匹配函数
def match_face(face_embedding):
    best_match = None
    highest_similarity = 0.6
    for name, saved_embedding in saved_face_embeddings.items():
        similarity = F.cosine_similarity(face_embedding, saved_embedding, dim=0).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = name
    return best_match

# 姿态关键点连接关系
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]

def draw_pose(image, keypoints, box_coords, client_socket):
    global is_tracking, last_sent_time
    now = time.time()

    if 'last_sent_time' not in globals():
        last_sent_time = now

    for i in range(17):
        x, y = keypoints[i]
        if x == 0 and y == 0:
            continue
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    for start, end in connections:
        start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
        end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
        if (start_point == (0, 0)) or (end_point == (0, 0)):
            continue
        cv2.line(image, start_point, end_point, (255, 0, 0), 2)

        # 抬手检测（启动跟踪）
        if (start == 5 and end == 7) or (start == 6 and end == 8):
            shoulder_y = keypoints[start][1]
            elbow_y = keypoints[end][1]
            if elbow_y < shoulder_y and not is_tracking:
                cv2.line(image, start_point, end_point, (0, 255, 255), 2)
                cv2.putText(image, "Hand Raised", (start_point[0], start_point[1] + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                send_command("hand_raised")
                is_tracking = True

    # 击掌检测
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    if np.any(left_wrist != (0, 0)) and np.any(right_wrist != (0, 0)):
        lx, ly = left_wrist
        rx, ry = right_wrist
        distance = np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry]))
        if distance < 100 and abs(ly - ry) < 50:
            mid_x = int((lx + rx) / 2)
            mid_y = int((ly + ry) / 2)
            cv2.putText(image, "Clap Detected", (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.line(image, (int(lx), int(ly)), (int(rx), int(ry)), (0, 0, 255), 3)
            send_command("clap")
            is_tracking = False

    # 发送坐标数据
    if is_tracking and box_coords is not None and now - last_sent_time > 0.5:
        x1, y1, x2, y2 = box_coords
        center_x = int((x1 + x2) / 2)
        width = int(x2 - x1)
        msg = f"{center_x},{width}"
        try:
            client_socket.sendall(msg.encode('utf-8'))
            last_sent_time = now
        except:
            print("Socket 连接异常")

# 加载 YOLOv8-Pose 模型
pose_model = YOLO('yolov8s-pose.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

recognized_name = None
face_present = False
is_tracking = False
last_sent_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取视频帧")
        break

    results = pose_model(frame)
    face_present = False

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            face = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            faces = mtcnn(face_pil)

            if faces is not None:
                face_present = True
                if recognized_name is None:
                    face_embedding = resnet(faces.unsqueeze(0)).squeeze(0)
                    matched_name = match_face(face_embedding)
                    if matched_name:
                        recognized_name = matched_name
                        print(f"识别成功：{recognized_name}")

            # 姿态识别前提：人脸识别成功且为授权人员
            if recognized_name in authorized_face_names:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                draw_pose(frame, keypoints[i].xy[0], box, client_socket)

    # UI 显示人脸识别状态
    if recognized_name and face_present:
        cv2.putText(frame, f"Recognized: {recognized_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif not face_present:
        recognized_name = None
        cv2.putText(frame, "等待人脸识别...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Pose and Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
