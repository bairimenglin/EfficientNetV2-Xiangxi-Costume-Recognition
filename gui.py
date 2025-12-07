import sys
import os
import json
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from torchvision import transforms
from model import efficientnetv2_s as create_model


class EthnicClothingRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("少数民族衣饰识别系统")
        self.setGeometry(100, 100, 1000, 1200)

        # 主窗口布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 标题
        self.title_label = QLabel("少数民族衣饰识别系统", self)
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 图片显示区域
        self.image_label = QLabel("上传的图片将在这里显示", self)
        self.image_label.setFixedSize(950, 800)
        self.image_label.setStyleSheet("background-color: gray; border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 上传按钮
        self.upload_button = QPushButton("上传图片", self)
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        # 分类结果显示
        self.result_label = QLabel("分类: \n置信度: ", self)
        self.result_label.setStyleSheet("font-size: 20px;")
        self.layout.addWidget(self.result_label)

        # 加载模型
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.class_indices = self.load_class_indices()

    def load_model(self):
        """加载训练好的模型权重"""
        model = create_model(num_classes=5).to(self.device)
        model_weight_path = "./weights/model-29.pth"  # 修改为实际权重路径
        assert os.path.exists(model_weight_path), f"模型权重文件 {model_weight_path} 不存在"
        model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        model.eval()
        return model

    def load_class_indices(self):
        """加载类别索引"""
        json_path = './class_indices.json'
        assert os.path.exists(json_path), f"类别索引文件 {json_path} 不存在"
        with open(json_path, "r") as f:
            class_indices = json.load(f)
        return class_indices

    def upload_image(self):
        """上传图片并进行预测"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            # 显示图片
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height())
            self.image_label.setPixmap(pixmap)

            # 进行预测
            result = self.predict(file_path)
            self.result_label.setText(result)

    def predict(self, img_path):
        """使用模型预测图片类别"""
        # 图像预处理
        img_size = {"s": [300, 384],  # train_size, val_size
                    "m": [384, 480],
                    "l": [384, 480]}
        num_model = "s"

        data_transform = transforms.Compose(
            [transforms.Resize(img_size[num_model][1]),
            transforms.CenterCrop(img_size[num_model][1]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 获取分类结果和置信度
        class_name = self.class_indices[str(predict_cla)]
        confidence = predict[predict_cla].item() * 100
        return f"分类: {class_name}\n置信度: {confidence:.2f}%"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EthnicClothingRecognitionApp()
    window.show()
    sys.exit(app.exec_())