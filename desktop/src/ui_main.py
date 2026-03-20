from __future__ import annotations

import numpy as np
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import (
    QWidget,
    QMainWindow,
    QLabel,
    QPushButton,
    QListWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
)
import pyqtgraph as pg

from ble_client import BleWorker
from config import DEFAULT_GAIN, DEFAULT_WINDOW_MS, DEFAULT_SAMPLE_RATE_HZ
from models import Sample
from processor import Processor


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BreathHeart Desktop")
        self.resize(1200, 800)

        self.processor = Processor(fs_hz=DEFAULT_SAMPLE_RATE_HZ)
        self.ble = BleWorker(sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ)

        self.raw_gain = float(DEFAULT_GAIN)
        self.raw_window_ms = int(DEFAULT_WINDOW_MS)

        self.scan_btn = QPushButton("Scan")
        self.stop_btn = QPushButton("Stop")
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.device_list = QListWidget()
        self.status_label = QLabel("Disconnected")

        self.hr_label = QLabel("Heart Rate: -- bpm")
        self.rr_label = QLabel("Respiration Rate: -- rpm")

        self.gain_down = QPushButton("-")
        self.gain_reset = QPushButton("Reset")
        self.gain_up = QPushButton("+")
        self.window_down = QPushButton("-")
        self.window_reset = QPushButton("Reset")
        self.window_up = QPushButton("+")
        self.gain_value = QLabel(f"x{self.raw_gain:.1f}")
        self.window_value = QLabel(f"{self.raw_window_ms/1000:.1f}s")

        self.raw_plot = pg.PlotWidget(title="Raw")
        self.resp_plot = pg.PlotWidget(title="Respiration")
        self.hr_plot = pg.PlotWidget(title="Heart Rate")
        for p in (self.raw_plot, self.resp_plot, self.hr_plot):
            p.showGrid(x=True, y=True, alpha=0.3)

        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#3b82f6", width=2))
        self.resp_curve = self.resp_plot.plot(pen=pg.mkPen("#10b981", width=2))
        self.hr_curve = self.hr_plot.plot(pen=pg.mkPen("#ef4444", width=2))

        self._build_ui()
        self._wire()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(33)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        left = QVBoxLayout()
        left.addWidget(QLabel("Devices"))
        left.addWidget(self.device_list, 1)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.stop_btn)
        left.addLayout(btn_row)
        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.connect_btn)
        btn_row2.addWidget(self.disconnect_btn)
        left.addLayout(btn_row2)
        left.addWidget(self.status_label)

        metrics = QHBoxLayout()
        metrics.addWidget(self.hr_label)
        metrics.addWidget(self.rr_label)

        raw_ctrl = QGridLayout()
        raw_ctrl.addWidget(QLabel("Gain"), 0, 0)
        raw_ctrl.addWidget(self.gain_down, 0, 1)
        raw_ctrl.addWidget(self.gain_reset, 0, 2)
        raw_ctrl.addWidget(self.gain_up, 0, 3)
        raw_ctrl.addWidget(self.gain_value, 0, 4)
        raw_ctrl.addWidget(QLabel("Window"), 1, 0)
        raw_ctrl.addWidget(self.window_down, 1, 1)
        raw_ctrl.addWidget(self.window_reset, 1, 2)
        raw_ctrl.addWidget(self.window_up, 1, 3)
        raw_ctrl.addWidget(self.window_value, 1, 4)

        raw_group = QGroupBox("Raw Waveform Controls")
        raw_group.setLayout(raw_ctrl)

        right = QVBoxLayout()
        right.addLayout(metrics)
        right.addWidget(raw_group)
        right.addWidget(self.raw_plot, 1)
        right.addWidget(self.resp_plot, 1)
        right.addWidget(self.hr_plot, 1)

        root = QHBoxLayout(central)
        root.addLayout(left, 1)
        root.addLayout(right, 3)

    def _wire(self) -> None:
        self.scan_btn.clicked.connect(self.ble.start_scan)
        self.stop_btn.clicked.connect(self.ble.stop_scan)
        self.connect_btn.clicked.connect(self._connect_selected)
        self.disconnect_btn.clicked.connect(self.ble.disconnect_device)

        self.gain_down.clicked.connect(self._gain_down)
        self.gain_up.clicked.connect(self._gain_up)
        self.gain_reset.clicked.connect(self._gain_reset)

        self.window_down.clicked.connect(self._window_down)
        self.window_up.clicked.connect(self._window_up)
        self.window_reset.clicked.connect(self._window_reset)

        self.ble.devices_updated.connect(self._update_devices)
        self.ble.connection_changed.connect(self._on_connection)
        self.ble.sample_received.connect(self._on_sample)
        self.ble.error.connect(self._on_error)

    def _update_devices(self, devices) -> None:
        self.device_list.clear()
        for d in devices:
            label = f"{d.name or 'Unknown'} ({d.address})"
            self.device_list.addItem(label)

    def _connect_selected(self) -> None:
        row = self.device_list.currentRow()
        if row < 0:
            return
        text = self.device_list.currentItem().text()
        address = text.split("(")[-1].strip(")")
        self.ble.connect_device(address)

    def _on_connection(self, ok: bool) -> None:
        self.status_label.setText("Connected" if ok else "Disconnected")
        if ok:
            self.processor.reset()

    def _on_error(self, msg: str) -> None:
        self.status_label.setText(msg)

    def _on_sample(self, sample: Sample) -> None:
        self.processor.on_sample(sample)

    def _update_plots(self) -> None:
        self._update_rates()
        self._plot_buffer(self.raw_curve, self.processor.raw_buf, self.raw_window_ms, self.raw_gain)
        self._plot_buffer(self.resp_curve, self.processor.resp_buf, self.raw_window_ms, 1.0)
        self._plot_buffer(self.hr_curve, self.processor.hr_buf, self.raw_window_ms, 1.0)

    def _update_rates(self) -> None:
        r = self.processor.rates
        self.hr_label.setText(
            f"Heart Rate: {r.bpm:.0f} bpm" if r.bpm is not None else "Heart Rate: -- bpm"
        )
        self.rr_label.setText(
            f"Respiration Rate: {r.rpm:.0f} rpm" if r.rpm is not None else "Respiration Rate: -- rpm"
        )

    def _plot_buffer(self, curve, buf, window_ms: int, gain: float) -> None:
        ts, vs = buf.snapshot()
        if not ts:
            curve.setData([])
            return
        t_max = ts[-1]
        t_min = t_max - window_ms
        idx = np.searchsorted(np.array(ts), t_min)
        t = np.array(ts[idx:], dtype=np.float64)
        v = np.array(vs[idx:], dtype=np.float64) * gain
        if t.size == 0:
            curve.setData([])
            return
        x = (t - t_min) / 1000.0
        curve.setData(x, v)

    def _gain_down(self) -> None:
        self.raw_gain = max(0.1, self.raw_gain / 1.2)
        self.gain_value.setText(f"x{self.raw_gain:.1f}")

    def _gain_up(self) -> None:
        self.raw_gain = min(200.0, self.raw_gain * 1.2)
        self.gain_value.setText(f"x{self.raw_gain:.1f}")

    def _gain_reset(self) -> None:
        self.raw_gain = 1.0
        self.gain_value.setText(f"x{self.raw_gain:.1f}")

    def _window_down(self) -> None:
        self.raw_window_ms = int(max(1000, self.raw_window_ms / 1.5))
        self.window_value.setText(f"{self.raw_window_ms/1000:.1f}s")

    def _window_up(self) -> None:
        self.raw_window_ms = int(min(30000, self.raw_window_ms * 1.5))
        self.window_value.setText(f"{self.raw_window_ms/1000:.1f}s")

    def _window_reset(self) -> None:
        self.raw_window_ms = 6000
        self.window_value.setText(f"{self.raw_window_ms/1000:.1f}s")
