import os
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

def calculate_mean_std(data_dir):
    """
    计算指定目录下所有图片的均值和标准差
    :param data_dir: 图片数据所在的根目录
    :return: mean, std
    """
    # 定义图像预处理
    transform = transforms.ToTensor()

    # 初始化变量
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0

    # 遍历所有图片
    for root, _, files in os.walk(data_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
                img_tensor = transform(img)  # 转换为Tensor格式

                # 累加像素值和像素平方值
                pixel_sum += img_tensor.sum(dim=(1, 2)).numpy()
                pixel_squared_sum += (img_tensor ** 2).sum(dim=(1, 2)).numpy()
                num_pixels += img_tensor.shape[1] * img_tensor.shape[2]

    # 计算均值和标准差
    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_squared_sum / num_pixels - mean ** 2)
    return mean, std


if __name__ == "__main__":
    data_dir = "./data"  # 修改为你的数据目录路径
    assert os.path.exists(data_dir), f"数据目录 {data_dir} 不存在"

    mean, std = calculate_mean_std(data_dir)
    print(f"Mean: {mean}")
    print(f"Std: {std}")