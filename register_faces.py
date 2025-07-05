import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 初始化 MTCNN（人脸检测）和 InceptionResnetV1（人脸识别）
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)  
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 存储人脸特征向量的字典
face_database = {}

# 指定人脸图片所在文件夹
face_folder = "faces"

# 确保目录存在
if not os.path.exists(face_folder):
    os.makedirs(face_folder)
    print(f"文件夹 '{face_folder}' 不存在，已创建！请放入人脸图片。")
    exit()

# 遍历 faces 文件夹中的所有图片
for filename in os.listdir(face_folder):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(face_folder, filename)
        image = Image.open(image_path)

        # 人脸检测
        face = mtcnn(image)

        if face is not None:
            # 计算人脸特征
            embedding = resnet(face.unsqueeze(0)).squeeze(0)
            # 存入字典，键为文件名，值为特征向量
            face_database[filename] = embedding
            print(f"录入人脸: {filename}")
        else:
            print(f"警告: {filename} 未检测到人脸，跳过！")

# 保存人脸特征数据
torch.save(face_database, "saved_faces.pth")
print("所有人脸数据已保存到 'saved_faces.pth' 文件")
