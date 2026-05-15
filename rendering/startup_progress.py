class StartupProgress:
    """Small Qt startup dialog for long scene preparation phases.

    The dialog is intentionally driven by explicit step calls from the main
    thread. It does not start a Qt event loop, which keeps it compatible with
    the pygame/OpenGL viewport that takes over once rendering begins.
    """

    def __init__(self, title="Roxy", total_steps=8):
        self._current = 0
        self._total = max(1, int(total_steps))
        self._app = None
        self._dialog = None
        self._headline = None
        self._detail = None
        self._bar = None
        self._qt_core = None
        self._enabled = self._init_qt(title)

    def step(self, headline, detail=""):
        self._current = min(self._current + 1, self._total)
        self.update(headline, detail)

    def update(self, headline, detail=""):
        if not self._enabled:
            print(f"[startup] {headline}")
            if detail:
                print(f"          {detail}")
            return

        self._headline.setText(headline)
        self._detail.setText(detail)
        self._bar.setValue(self._current)
        self._dialog.adjustSize()
        self._dialog.show()
        self._app.processEvents(
            self._qt_core.QEventLoop.ProcessEventsFlag.AllEvents,
            50,
        )

    def close(self):
        if not self._enabled:
            return
        self._dialog.close()
        self._app.processEvents(
            self._qt_core.QEventLoop.ProcessEventsFlag.AllEvents,
            50,
        )

    def _init_qt(self, title):
        try:
            from PySide6 import QtCore, QtWidgets
        except Exception:
            return False

        self._qt_core = QtCore
        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(title)
        dialog.setModal(False)
        dialog.setMinimumWidth(380)
        dialog.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        dialog.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        headline = QtWidgets.QLabel("Preparing scene")
        font = headline.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        headline.setFont(font)

        detail = QtWidgets.QLabel("")
        detail.setWordWrap(True)
        detail.setMinimumHeight(36)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, self._total)
        bar.setValue(0)
        bar.setTextVisible(True)

        layout.addWidget(headline)
        layout.addWidget(detail)
        layout.addWidget(bar)

        self._dialog = dialog
        self._headline = headline
        self._detail = detail
        self._bar = bar
        return True
