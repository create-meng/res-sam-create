# -*- coding: utf-8 -*-
from ui.utils import *


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setup()

        self.setWindowTitle(i18n("GPR 地下灾害探测"))
        self.base_layer = None

        self.lock = False

        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _features_path = os.path.join(_repo_root, "PatchRes", "features.pth")
        self.tile_res = PatchRes(stride=10, anomaly_threshold=0.1, hidden_size=30, window_size=[50, 50], features=_features_path)

        self.tile_res.fit(0)
        self.heat_map = None

    def setup(self):
        self.setFixedSize(684, 559)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignLeft)

    def draw_base_layer(self):
        if self.base_layer is None:
            self.image_label.clear()
            self.lock = False
            return
        final_pixmap = QPixmap(self.base_layer.size())
        final_pixmap.fill(Qt.transparent)

        painter = QPainter(final_pixmap)

        painter.drawPixmap(0, 0, self.base_layer)
        painter.end()
        self.image_label.setPixmap(final_pixmap)

    def undo(self):
        if self.lock:
            return
        self.heat_map = None
        self.draw_base_layer()

    def reset(self):
        if self.lock:
            return
        self.heat_map = None
        self.draw_base_layer()

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, i18n("选择图片"), "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
        if file_name:
            self.heat_map = None
            self.base_layer = QPixmap(file_name).scaled(self.image_label.size(), Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation)
            self.draw_base_layer()

    def get_image(self):
        if self.base_layer is None:
            return
        image = self.base_layer.toImage()
        width = image.width()
        height = image.height()
        ptr = image.bits()
        arr = np.array(ptr).reshape((height, width, 4))

        arr = arr[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (arr - arr.mean()) / arr.std()

        return arr
    
    def loading(self):
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.show_centered_in(self.image_label)

    def get_mask(self, image, x0, y0, x1, y1):
        image = torch.from_numpy(image).float()
        area = image[y0: y1, x0: x1]
        mask, _, _, _, _ = self.tile_res.predict(torch.mean(
            area.permute(2, 0, 1), dim=0, keepdim=True), mode="pixel_seg")
        mask = torch.from_numpy(mask)
        new_mask = torch.zeros_like(image)

        # print(mask.shape)
        mask = mask.squeeze(0)
        new_mask[y0: y1, x0: x1] = torch.cat(
            [mask[0:y1 - y0, :x1 - x0].unsqueeze(2)] * 3, dim=2)

        # self.draw_rect(box)
        new_mask = torch.cat([mask[0:y1 - y0, :x1 - x0].unsqueeze(2)] * 3, dim=2)

        new_mask = new_mask.numpy() * 255
        new_mask = new_mask.astype(np.uint8)
        return new_mask

    def draw_rect(self, box, color="red"):
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        updated_pixmap = pixmap.copy()
        painter = QPainter(updated_pixmap)
        pen = QPen(QColor(color), 3)
        painter.setPen(pen)

        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0

        painter.drawRect(x0, y0, width, height)
        painter.end()

        self.image_label.setPixmap(updated_pixmap)

    def gen_full_heatmap(self):
        self.loading()
        self._hm_thread = QThread(self)
        self._hm_worker = GetMaskWithScoreWorker(self.get_image, self.get_mask, self.tile_res.extractor.fit_without_tiling, self.tile_res.anomaly_scorer.predict)
        self._hm_worker.moveToThread(self._hm_thread)

        self._hm_thread.started.connect(self._hm_worker.run)
        self._hm_worker.finished.connect(self._on_heatmap_ready)
        self._hm_worker.failed.connect(self._on_heatmap_failed)

        self._hm_worker.finished.connect(self._hm_thread.quit)
        self._hm_worker.finished.connect(self._hm_worker.deleteLater)
        self._hm_thread.finished.connect(self._hm_thread.deleteLater)

        self._hm_worker.failed.connect(self._hm_thread.quit)
        self._hm_worker.failed.connect(self._hm_worker.deleteLater)

        self._hm_thread.start()

    @Slot(object, int, int)
    def _on_heatmap_ready(self, mask, x0, y0):
        self.heat_map = (mask, x0, y0)
        self.draw_mask(mask, x0, y0)
        self._continue_after_heatmap()

    @Slot(str)
    def _on_heatmap_failed(self, message):
        try:
            QMessageBox.warning(self, i18n("提示"), message)
        finally:
            if hasattr(self, "loading_overlay") and self.loading_overlay is not None:
                self.loading_overlay.close()

    def _continue_after_heatmap(self):
        heat_map, x, y = self.heat_map
        heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())

        anomaly_threshold = 0.1
        foreground_pixels = np.argwhere(heat_map > anomaly_threshold)
        # print(foreground_pixels)
        # print(heat_map.shape) # 500,500,3
        if foreground_pixels.size == 0:
            min_row, min_col, max_row, max_col = 0, 0, 0, 0
        else:
            min_row = int(
                np.min(foreground_pixels[:, 0]) + y)
            min_col = int(
                np.min(foreground_pixels[:, 1]) + x)
            max_row = int(
                np.max(foreground_pixels[:, 0]) + y)
            max_col = int(
                np.max(foreground_pixels[:, 1]) + x)
        box = (min_col, min_row, max_col, max_row)
        self.draw_rect(box)
        self.loading_overlay.close()

    def output_box(self):
        self.gen_full_heatmap()

    def save(self):
        pixmap = self.image_label.pixmap()
        success = False
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(self, i18n("保存图片到文件"), "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                if pixmap.save(file_path):
                    success = True
        if success:
            QMessageBox.information(self, i18n("保存成功"), i18n("图片已保存到") + f"\n{file_path}")
        else:
            QMessageBox.warning(self, i18n("保存失败"), i18n("图片保存失败"))

    def draw_mask(self, mask, x, y):
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        updated_pixmap = pixmap.copy()
        painter = QPainter(updated_pixmap)
        painter.setOpacity(0.4)
        mask = cv2.applyColorMap(mask[:, :, 0], cv2.COLORMAP_JET)
        # mask = 255 - mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = QImage(
            mask.data, mask.shape[1], mask.shape[0], mask.shape[1] * 3, QImage.Format_RGB888)

        painter.drawImage(x, y, image)
        painter.end()

        self.image_label.setPixmap(updated_pixmap)
