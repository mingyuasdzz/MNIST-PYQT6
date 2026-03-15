import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QProgressBar)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPen, QImage, QColor, QFont
import torch
from torchvision import transforms
from cnn_bn import Net, load_model

class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setMaximumSize(280, 280)
        self.image = QImage(280, 280, QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 15

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.GlobalColor.black, self.brush_size,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear_canvas(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def get_tensor(self):
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width)
        rows = np.any(arr < 128, axis=1)
        cols = np.any(arr < 128, axis=0)

        if not rows.any() or not cols.any():
            arr_centered = np.full((280, 280), 255, dtype=np.uint8)
        else:
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            y_padding = int((y_max - y_min) * 0.12)
            x_padding = int((x_max - x_min) * 0.12)
            y_min = max(0, y_min - y_padding)
            y_max = min(height, y_max + y_padding)
            x_min = max(0, x_min - x_padding)
            x_max = min(width, x_max + x_padding)
            digit_region = arr[y_min:y_max, x_min:x_max]
            digit_height, digit_width = digit_region.shape
            arr_centered = np.full((280, 280), 255, dtype=np.uint8)
            y_start = (280 - digit_height) // 2
            x_start = (280 - digit_width) // 2
            arr_centered[y_start:y_start+digit_height, x_start:x_start+digit_width] = digit_region

        from PIL import Image
        img_centered = Image.fromarray(arr_centered, mode='L')

        pil_img = img_centered.resize((28, 28), Image.Resampling.LANCZOS)
        arr_resized = np.array(pil_img)

        pil_img = Image.fromarray(255 - arr_resized, mode='L')
        tensor = transforms.Compose([transforms.ToTensor()])(pil_img)
        tensor = tensor.unsqueeze(0)

        return tensor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('手写数字识别')
        self.setGeometry(100, 100, 500, 350)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Net().to(self.device)
        try:
            self.model.load_state_dict(torch.load('model.pth', map_location=self.device), strict=True)
            self.model.eval()
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        self.canvas = DrawingCanvas()
        left_layout.addWidget(self.canvas)

        canvas_label = QLabel('在此处手写数字 (280x280)')
        canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(canvas_label)

        button_layout = QHBoxLayout()
        left_layout.addLayout(button_layout)

        clear_btn = QPushButton('清空')
        clear_btn.clicked.connect(self.canvas.clear_canvas)
        button_layout.addWidget(clear_btn)

        recognize_btn = QPushButton('识别')
        recognize_btn.clicked.connect(self.recognize_digit)
        button_layout.addWidget(recognize_btn)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        result_label = QLabel('识别结果')
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_label.setFont(QFont('Arial', 16))
        right_layout.addWidget(result_label)

        self.result_display = QLabel('?')
        self.result_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_display.setFont(QFont('Arial', 72))
        self.result_display.setStyleSheet('color: #4CAF50; font-weight: bold;')
        right_layout.addWidget(self.result_display)

        self.confidence_label = QLabel('置信度: --')
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setFont(QFont('Arial', 12))
        right_layout.addWidget(self.confidence_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet('QProgressBar::chunk { background-color: #4CAF50; }')
        right_layout.addWidget(self.confidence_bar)

        self.top3_label = QLabel('Top-3 预测:')
        self.top3_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3_label.setFont(QFont('Arial', 10))
        right_layout.addWidget(self.top3_label)

        right_layout.addStretch()

    def recognize_digit(self):
        tensor = self.canvas.get_tensor().to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            top3_probs, top3_indices = torch.topk(probabilities, 3)

        predicted = top3_indices[0].item()
        confidence = top3_probs[0].item() * 100

        self.result_display.setText(str(predicted))
        self.confidence_label.setText(f'置信度: {confidence:.2f}%')
        self.confidence_bar.setValue(int(confidence))

        top3_text = 'Top-3 预测:\n'
        for i in range(3):
            digit = top3_indices[i].item()
            prob = top3_probs[i].item() * 100
            top3_text += f'{digit}: {prob:.2f}%\n'
        self.top3_label.setText(top3_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())