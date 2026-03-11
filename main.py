import sys
from PySide6.QtWidgets import QApplication
from ui.ui_funcs import MainWindow
import os


os.environ['LANGUAGE'] = 'zh_CN' # en / zh_CN


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
