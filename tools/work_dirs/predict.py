import os
import shutil
from mmpretrain import ImageClassificationInferencer

# 配置和检查点路径
config = '/home/xlsk/Code/mmpretrain/configs/resnet/resnet50_8xb32_in1k_mine.py'
checkpoint = '/home/xlsk/Code/mmpretrain/tools/work_dirs/resnet50_8xb32_in1k_mine/epoch_69.pth'

# 创建推理器
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')

# 获取imagenet文件夹中的所有图片
image_folder = 'new_data'
image_list = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 执行推理
results = inferencer(image_list, batch_size=8)

# 创建预测结果文件夹，带编号
def create_numbered_folder(base_name):
    idx = 1
    while True:
        folder_name = f"{base_name}{idx}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        idx += 1

predict_folder_base = create_numbered_folder('predict')

# 处理并保存结果
for idx, result in enumerate(results):
    pred_class = result['pred_class']
    image_path = image_list[idx]

    # 创建类别文件夹
    class_folder = os.path.join(predict_folder_base, pred_class)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    # 复制图片到类别文件夹
    image_name = os.path.basename(image_path)
    dest_path = os.path.join(class_folder, image_name)
    shutil.copy(image_path, dest_path)

print(f"预测结果已保存到 {predict_folder_base} 文件夹中。")

