import timm
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import json
import urllib.request
import googletrans
from wand.image import Image as WandImage
from io import BytesIO

def translate_text(text, target_language='zh-tw'):
    translator = googletrans.Translator()
    t = translator.translate(text, dest=target_language).text
    return t

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# 初始化模型
models = {
    'vit': timm.create_model('vit_large_patch16_224', pretrained=True),
    'swin': timm.create_model('swin_large_patch4_window7_224', pretrained=True),
    'deit': timm.create_model('deit_base_patch16_224', pretrained=True),
    'convnext': timm.create_model('convnext_large', pretrained=True),
    'efficientnetv2': timm.create_model('tf_efficientnetv2_l', pretrained=True),
    'bit': timm.create_model('resnetv2_152x2_bit.goog_in21k_ft_in1k', pretrained=True),
    'mobilenetv3': timm.create_model('mobilenetv3_large_100_miil', pretrained=True),
    'densenet': timm.create_model('densenet121', pretrained=True),
    'regnet': timm.create_model('regnetx_032', pretrained=True),
    'resnext': timm.create_model('resnext50_32x4d', pretrained=True),
    'inceptionv3': timm.create_model('tf_inception_v3', pretrained=True)
}

# 将所有模型移动到相同的设备
for model in models.values():
    model.to(device)
    model.eval()

def download_imagenet_labels():
    while True:
        try:
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            response = urllib.request.urlopen(url)
            labels = json.loads(response.read())
            return labels
        except:
            continue

def read_image(image_path):
    """
    读取图像文件，支持常规格式和 HEIC 格式。
    如果是 HEIC 格式，则在内存中转换为 JPEG。
    """
    if image_path.lower().endswith(".heic"):
        with WandImage(filename=image_path) as img:
            with img.convert('jpeg') as converted:
                jpeg_bytes = converted.make_blob('jpeg')
        pil_image = Image.open(BytesIO(jpeg_bytes))
    else:
        pil_image = Image.open(image_path)

    return pil_image

# 其余的代码（add_tags_to_filename, rename_files_in_folder, main）保持不变
def process_image(image_path):
    pil_image = read_image(image_path)
    image_array = np.array(pil_image)

    # 确保图像是三通道的 RGB
    if image_array.shape[-1] == 4:  # 如果有 4 个通道（RGBA），则转换为 RGB
        image_array = image_array[:, :, :3]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_array).unsqueeze(0).to(device)

def recognize_image(image_path):
    try:
        img_tensor = process_image(image_path)
    except:
        return 0

    label_counts = {}

    # 对每个模型进行预测并存储结果
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top7_prob, top7_catid = torch.topk(probabilities, 7)
        for catid in top7_catid[0]:
            label = imagenet_labels[catid.item()]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

    # 按重复次数对标签排序
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)

    # 选择前 7 个最常出现的标签
    final_labels = sorted_labels[:7]

    # 翻译标签
    translated_labels = []
    for label in final_labels:
        try:
            t = translate_text(label)
            translated_labels.append(t)
        except:
            continue

    return translated_labels

# 将标签添加到文件名
def is_valid_tag(tag):
    invalid_chars = set('<>:"/\\|?*\t')
    return not any((c in invalid_chars) for c in tag)

def add_tags_to_filename(file_path, tags):
    # 过滤掉包含不合法字符的标签
    valid_tags = [tag for tag in tags if is_valid_tag(tag)]

    if not valid_tags:
        print(f"所有標籤不合法，跳過{file_path}")
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
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".heic")):
                        image_path = os.path.join(folder_path, filename)
                        tags = recognize_image(image_path)
                        if tags != 0:
                            print("已處理", image_path)
                            print(tags)
                            add_tags_to_filename(image_path, tags)
                        else:
                            print("讀取失敗，跳過", image_path)
                            continue

# 指定根目录路径
root_folder_path = input("照片資料夾路徑：")
imagenet_labels = download_imagenet_labels()
main(root_folder_path)