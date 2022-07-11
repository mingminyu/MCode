import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from FirstDesktopApplicationWithQtDesigner import Ui_MainWindow


if __name__ == '__main__':
    # 创建 QApplication 类的实例
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    # 向主窗口上添加空间
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
