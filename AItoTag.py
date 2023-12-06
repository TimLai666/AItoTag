import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from translate import Translator
import json
import urllib.request

# 使用新的weights参数加载预训练的 ResNet50 模型
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

def process_image(image_path):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    image_array = image_array[:, :, ::-1].copy()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_array).unsqueeze(0)

def translate_text_with_translate_lib(text, target_language='zh-TW'):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

# 从网络上下载 ImageNet 类别标签
def download_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = urllib.request.urlopen(url)
    labels = json.loads(response.read())
    return labels

def recognize_image(image_path):
    img_tensor = process_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_labels = [imagenet_labels[catid.item()] for catid in top5_catid]
    # 翻译标签
    translated_labels = [translate_text_with_translate_lib(label) for label in top5_labels]
    return translated_labels

# 将标签添加到文件名
def add_tags_to_filename(file_path, tags):
    if not tags:
        return  # 如果没有标签，则不进行操作
    name, ext = os.path.splitext(file_path)
    new_name = f"{name}_ait_o{'_'.join(tags)}{ext}"
    os.rename(file_path, new_name)

def rename_files_in_folder(folder_path):
    for folder_path, _, filenames in os.walk(root_folder_path):
        for filename in filenames:
            # 检查文件名中是否包含特定字符串 "_ait_o"
            if "_ait_o" in filename:
                # 找到 "_ait_o" 字符串的位置
                index = filename.find("_ait_o")
                # 获取新的文件名，即去除 "_ait_o" 及其后面的部分
                new_filename = filename[:index] + os.path.splitext(filename)[1]
                # 构建完整的旧文件路径和新文件路径
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_filename}'")

# 主函数
def main(root_folder_path):
    f = input("選擇功能，\n1：自動辨識，2：移除標籤：")
    if f == "2":
        rename_files_in_folder(root_folder_path)
    elif f == "1":
        for folder_path, _, filenames in os.walk(root_folder_path):
            for filename in filenames:
                if "_ait_o" in filename:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(folder_path, filename)
                        tags = recognize_image(image_path)
                        add_tags_to_filename(image_path, tags)

# 指定根目录路径
root_folder_path = input("照片資料夾路徑：")
imagenet_labels = download_imagenet_labels()
main(root_folder_path)
