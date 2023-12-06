import timm
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from translate import Translator
import json
import urllib.request

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# 使用 EfficientNet
model_name = 'efficientnet_b5'  # 您可以选择不同的版本，例如 'efficientnet_b0' 到 'efficientnet_b7'
model = timm.create_model(model_name, pretrained=True)
model.to(device)  # 确保模型也在 GPU 上
model.eval()

# 其余的函数（process_image, translate_text_with_translate_lib 等）保持不变

def recognize_image(image_path):
    img_tensor = process_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)  # 输入数据现在在 GPU 上
        print(outputs)
    # 获取 top-5 类别
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(top5_prob, top5_catid)
    # 将类别 ID 转换为自然语言标签
    top5_labels = [imagenet_labels[catid.item()] for catid in top5_catid[0]]
    print(top5_labels)
    # 翻译标签
    translated_labels = [translate_text_with_translate_lib(label) for label in top5_labels]
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
                if not "_ait_o" in filename:
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(folder_path, filename)
                        tags = recognize_image(image_path)
                        add_tags_to_filename(image_path, tags)

# 指定根目录路径
root_folder_path = input("照片資料夾路徑：")
imagenet_labels = download_imagenet_labels()
main(root_folder_path)