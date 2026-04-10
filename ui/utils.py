# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QMainWindow, QFileDialog, QLabel, QWidget, QVBoxLayout, QMessageBox
from PySide6.QtGui import QFontMetrics, QPixmap, QPainter, QImage, QPen, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
import numpy as np
from io import BytesIO
import cv2
import torch
from PIL import ImageQt, Image
from ui.ui_main import Ui_MainWindow
from sam.sam import predict_mask
from PatchRes.PatchRes import PatchRes
from ui.i18n import i18n

class GetMaskWithScoreWorker(QObject):
    finished = Signal(object, int, int)   # mask, x0, y0
    failed = Signal(str)

    def __init__(self, get_image_fn, get_mask_fn, extract_feature_fn, score_predict_fn=None):
        super().__init__()
        self.get_image = get_image_fn
        self.get_mask_fn = get_mask_fn
        self.extract_feature_fn = extract_feature_fn
        self.score_predict_fn = score_predict_fn

    @Slot()
    def run(self):
        image = self.get_image()
        mask_boxes = predict_mask(image)
        scores = []
        boxes = []
        if mask_boxes is None or len(mask_boxes) == 0:
            self.failed.emit(i18n("未检测到候选区域"))
            return
        for box in mask_boxes:
            try:
                x0, y0, x1, y1 = box
                feature = self.extract_feature_fn(torch.from_numpy(image[y0:y1, x0:x1,0]).float().unsqueeze(0))
                score = self.score_predict_fn(feature)[0]
                scores.append(score)
                boxes.append(box)
            except:
                ...
        
        if len(scores) == 0:
            self.failed.emit(i18n("候选区域评分失败"))
            return

        scores = np.array(scores).squeeze()
        is_abnormal = np.where(scores > 0.1)[0]
        abnormal_boxes = []
        
        for i in is_abnormal:
            abnormal_boxes.append(boxes[i])
        if len(abnormal_boxes) == 0:
            self.failed.emit(i18n("未检测到异常区域"))
            return
        abnormal_boxes = np.array(abnormal_boxes)
        x_m = abnormal_boxes[:, 0].min()
        y_m = abnormal_boxes[:, 1].min()
        x_M = abnormal_boxes[:, 2].max()
        y_M = abnormal_boxes[:, 3].max()

        mask = self.get_mask_fn(image, x_m, y_m, x_M, y_M)
        self.finished.emit(mask, x_m, y_m)
        return boxes


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 10px;")
        
        label = QLabel(i18n("正在绘制，请稍候..."), self)
        label.setStyleSheet("color: white; font-size: 16px;")
        label.setAlignment(Qt.AlignCenter)

        self.label = label
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)

    def show_centered_in(self, parent_widget):
        font_metrics = QFontMetrics(self.label.font())
        text_width = font_metrics.horizontalAdvance(self.label.text())

        parent_rect = parent_widget.geometry()
        overlay_width = text_width + 200
        overlay_height = 100
        x = parent_rect.x() + (parent_rect.width() - overlay_width) // 2
        y = parent_rect.y() + (parent_rect.height() - overlay_height) // 2

        self.setGeometry(x, y, overlay_width, overlay_height)
        self.show()
