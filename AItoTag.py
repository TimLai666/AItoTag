import timm
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from translate import Translator
import json
import urllib.request
from easynmt import EasyNMT
from easynmt import EasyNMT
import zhconv

def translate_text_with_easynmt(text, target_language='zh'):
    return zhconv.convert(translate_model.translate(text, target_lang=target_language), 'zh-tw')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

translate_model = EasyNMT('m2m_100_1.2b')
# 使用 Vision Transformer
model_name = 'vit_large_patch16_224'  # 例如 'vit_base_patch16_224', 'vit_large_patch16_224' 等
model = timm.create_model(model_name, pretrained=True)
model.to(device)  # 确保模型也在 GPU 上
model.eval()

def download_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = urllib.request.urlopen(url)
    labels = json.loads(response.read())
    return labels

def recognize_image(image_path):
    img_tensor = process_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    top5_labels = [imagenet_labels[catid.item()] for catid in top5_catid[0]]
    # 使用 EasyNMT 进行翻译
    translated_labels = [translate_text_with_easynmt(label) for label in top5_labels]
    return translated_labels

# 其余的代码（add_tags_to_filename, rename_files_in_folder, main）保持不变
def process_image(image_path):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image_array).unsqueeze(0)
    return img_tensor.to(device)  # 将图像数据转移到 GPU

# 将标签添加到文件名
def is_valid_tag(tag):
    invalid_chars = set('<>:"/\\|?*\t')
    return not any((c in invalid_chars) for c in tag)

def add_tags_to_filename(file_path, tags):
    # 过滤掉包含不合法字符的标签
    valid_tags = [tag for tag in tags if is_valid_tag(tag)]

    if not valid_tags:
        print(f"所有标签不合法，跳过文件：{file_path}")
        return  # 如果没有合法的标签，则跳过该文件

    name, ext = os.path.splitext(file_path)
    new_name = f"{name}_ait_o{'_'.join(valid_tags)}{ext}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_name)
    os.rename(file_path, new_file_path)

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
                if not "_ait_o" in filename:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(folder_path, filename)
                        tags = recognize_image(image_path)
                        add_tags_to_filename(image_path, tags)

# 指定根目录路径
root_folder_path = input("照片資料夾路徑：")
imagenet_labels = download_imagenet_labels()
main(root_folder_path)