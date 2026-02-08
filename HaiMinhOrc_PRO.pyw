import os
import sys
import subprocess
import json
import datetime
import warnings
import logging
import shutil
import math
import queue
import time
from difflib import SequenceMatcher

# --- CẤU HÌNH PORTABLE & ĐƯỜNG DẪN GỐC ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_DIR = os.path.join(BASE_DIR, "SubTool_Temp")

# --- STYLE SHEET ---
MODERN_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
QGroupBox {
    border: 2px solid #3a3a3a;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: bold;
    color: #00d2ff;
    background-color: #252526;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
    background-color: #1e1e1e;
}
QLabel { color: #cfcfcf; font-size: 13px; }
/* LIVE MONITOR TEXT */
#lblLiveText {
    font-size: 16px;
    font-weight: bold;
    color: #ffd700;
    background-color: #000;
    border: 1px solid #555;
    border-radius: 5px;
    padding: 10px;
}
QComboBox {
    background-color: #333333;
    border: 1px solid #555;
    border-radius: 5px;
    padding: 5px;
    color: white;
    min-width: 100px;
}
QComboBox::drop-down { border-left-width: 0px; }
QTableWidget {
    background-color: #2d2d30;
    gridline-color: #3e3e42;
    color: #f0f0f0;
    border: none;
    border-radius: 5px;
}
QHeaderView::section {
    background-color: #3e3e42;
    padding: 4px;
    border: 1px solid #2d2d30;
    color: #00d2ff;
    font-weight: bold;
}
QTableWidget::item:selected {
    background-color: #0078d7;
    color: white;
}
QPushButton {
    border-radius: 8px;
    padding: 10px 15px;
    font-weight: bold;
    font-size: 14px;
    color: white;
    border: none;
}
QPushButton:disabled { background-color: #444; color: #888; }
#btnLoad { background-color: #444; border: 1px solid #666; }
#btnLoad:hover { background-color: #555; }
#btnStart { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #11998e, stop:1 #38ef7d); }
#btnStart:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #16a085, stop:1 #55efc4); margin-top: -2px; }
#btnStop { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #cb2d3e, stop:1 #ef473a); }
#btnStop:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e52d27, stop:1 #b31217); margin-top: -2px; }
#btnSave { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8E2DE2, stop:1 #4A00E0); }
#btnSave:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #9D50BB, stop:1 #6E48AA); margin-top: -2px; }
QSlider::groove:horizontal { border: 1px solid #3a3a3a; height: 8px; background: #202020; margin: 2px 0; border-radius: 4px; }
QSlider::handle:horizontal { background: #00d2ff; border: 1px solid #00d2ff; width: 18px; height: 18px; margin: -5px 0; border-radius: 9px; }
QCheckBox { color: #00d2ff; font-weight: bold; spacing: 5px; }
QCheckBox::indicator { width: 18px; height: 18px; }
"""

# --- CẤU HÌNH MÔI TRƯỜNG ---
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['FLAGS_use_mkldnn'] = '0' 
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

warnings.filterwarnings("ignore")
logging.getLogger("ppocr").setLevel(logging.ERROR)

import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QSlider, QMessageBox, QComboBox, QGroupBox, QFrame, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush

try:
    import paddle
    from paddleocr import PaddleOCR
except ImportError:
    pass

# --- HÀM HỆ THỐNG ---
def setup_temp_folder():
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR)
        except: pass
    try: os.makedirs(TEMP_DIR, exist_ok=True)
    except: pass

def cleanup_temp_folder():
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR)
        except: pass

def get_ffmpeg_path():
    local_ffmpeg = os.path.join(BASE_DIR, "ffmpeg.exe")
    if os.path.exists(local_ffmpeg): return local_ffmpeg
    if shutil.which("ffmpeg"): return "ffmpeg"
    return None

def get_ffprobe_path():
    local_ffprobe = os.path.join(BASE_DIR, "ffprobe.exe")
    if os.path.exists(local_ffprobe): return local_ffprobe
    if shutil.which("ffprobe"): return "ffprobe"
    if os.path.exists("ffprobe.exe"): return "ffprobe.exe"
    return None

def get_video_info(video_path):
    ffprobe = get_ffprobe_path()
    if not ffprobe: return 0, 0, 0, 0
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration:stream_tags=rotate",
        "-of", "json", video_path
    ]
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=creation_flags)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        try: duration = float(info.get('format', {}).get('duration', stream.get('duration', 0)))
        except: duration = 0
        return width, height, duration
    except: return 0, 0, 0, 0

def get_preview_frame(video_path, time_sec):
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe: return None, "Không tìm thấy file ffmpeg.exe!"
    cmd = [ffmpeg_exe, '-y', '-ss', str(time_sec), '-i', video_path, '-frames:v', '1', '-f', 'image2', '-vcodec', 'png', '-']
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8, creationflags=creation_flags)
        data, err = pipe.communicate()
        if not data: return None, "Lỗi đọc ảnh"
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img, None
    except Exception as e: return None, str(e)

# --- THREADS ---
class VideoCaptureThread(QThread):
    def __init__(self, video_path, roi, fps_scan, frame_queue):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.fps_scan = fps_scan
        self.queue = frame_queue
        self.is_running = True

    def run(self):
        w_vid, h_vid, _ = get_video_info(self.video_path)
        x, y, w, h = self.roi
        x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
        w, h = (w // 2) * 2, (h // 2) * 2
        if w < 2 or h < 2: return 
        ffmpeg_exe = get_ffmpeg_path()
        vf_filter = f"crop={w}:{h}:{x}:{y},fps={self.fps_scan}"
        cmd = [ffmpeg_exe, '-y', '-loglevel', 'error', '-i', self.video_path, '-vf', vf_filter, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-']
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8, creationflags=creation_flags)
        frame_size = w * h * 3
        frame_idx = 0
        while self.is_running:
            in_bytes = process.stdout.read(frame_size)
            if len(in_bytes) != frame_size: break
            frame = np.frombuffer(in_bytes, np.uint8).reshape((h, w, 3))
            try:
                self.queue.put((frame_idx, frame.copy()), timeout=1)
            except queue.Full: pass
            frame_idx += 1
        process.terminate()
        self.queue.put((None, None))
    def stop(self): self.is_running = False

class OCRWorker(QThread):
    update_signal = pyqtSignal(str, str, str)
    live_signal = pyqtSignal(QImage, str) 
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, roi, use_gpu, lang='ch', fps_scan=5, live_mode=False):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.use_gpu = use_gpu
        self.lang = lang
        self.fps_scan = fps_scan
        self.live_mode = live_mode
        self.is_running = True
        self.frame_queue = queue.Queue(maxsize=100)
        self.capture_thread = None

    def run(self):
        try:
            if self.use_gpu:
                try: paddle.device.set_device('gpu')
                except: paddle.device.set_device('cpu')
            else: paddle.device.set_device('cpu')
            
            vid_name = os.path.basename(self.video_path)
            self.log_signal.emit(f"[{vid_name}] Đang nạp Model...")
            
            ocr = PaddleOCR(use_angle_cls=False, lang=self.lang, show_log=False, enable_mkldnn=True, ocr_version='PP-OCRv3')
            _, _, duration_vid = get_video_info(self.video_path)

            self.log_signal.emit(f"[{vid_name}] Bắt đầu quét...")
            self.capture_thread = VideoCaptureThread(self.video_path, self.roi, self.fps_scan, self.frame_queue)
            self.capture_thread.start()

            current_sub = None
            
            # --- CẤU HÌNH BỘ LỌC ỔN ĐỊNH ---
            last_raw_text = ""
            stability_count = 0
            STABLE_THRESHOLD = 2
            consecutive_lost = 0
            LOST_THRESHOLD = 3

            while self.is_running:
                try: 
                    frame_idx, frame = self.frame_queue.get(timeout=5)
                except queue.Empty:
                    if self.capture_thread.isRunning(): continue
                    else: break
                if frame_idx is None: break

                # OCR
                result = ocr.ocr(frame, cls=False)
                raw_text_found = ""
                if result and result[0]:
                    valid_lines = [line[1][0] for line in result[0] if line[1][1] > 0.6]
                    raw_text_found = " ".join(valid_lines).strip()

                # --- BƯỚC 1: LỌC NHIỄU (DEBOUNCE) ---
                final_text = ""
                if raw_text_found:
                    if raw_text_found == last_raw_text:
                        stability_count += 1
                    else:
                        last_raw_text = raw_text_found
                        stability_count = 1
                    
                    if stability_count >= STABLE_THRESHOLD:
                        final_text = raw_text_found
                else:
                    last_raw_text = ""
                    stability_count = 0

                # Live Monitor
                if self.live_mode:
                    h, w, ch = frame.shape
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    display_text = raw_text_found if raw_text_found else "..."
                    self.live_signal.emit(qt_img, display_text)

                timestamp_sec = frame_idx * (1.0 / self.fps_scan)

                # --- BƯỚC 2: LOGIC TẠO SUB ---
                if final_text:
                    consecutive_lost = 0 
                    if current_sub is None:
                        current_sub = {'start': timestamp_sec, 'end': timestamp_sec, 'text': final_text}
                    else:
                        similarity = SequenceMatcher(None, current_sub['text'], final_text).ratio()
                        if similarity > 0.8: 
                            current_sub['end'] = timestamp_sec
                            if len(final_text) > len(current_sub['text']): 
                                current_sub['text'] = final_text
                        else:
                            self.emit_sub(current_sub)
                            current_sub = {'start': timestamp_sec, 'end': timestamp_sec, 'text': final_text}
                else:
                    if current_sub:
                        consecutive_lost += 1
                        if consecutive_lost >= LOST_THRESHOLD:
                            self.emit_sub(current_sub)
                            current_sub = None
                            consecutive_lost = 0

                if frame_idx % 15 == 0 and duration_vid > 0:
                    percent = int((timestamp_sec / duration_vid) * 100)
                    self.progress_signal.emit(percent)

            if current_sub: self.emit_sub(current_sub)
            if self.capture_thread.isRunning():
                self.capture_thread.stop()
                self.capture_thread.wait()
            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"Lỗi: {str(e)}")
            self.finished_signal.emit()

    def emit_sub(self, sub_data):
        def format_time(seconds):
            total_sec = int(seconds)
            ms = int((seconds - total_sec) * 1000)
            base = str(datetime.timedelta(seconds=total_sec))
            if len(base.split(":")) < 3: base = "0" + base 
            return f"{base},{ms:03d}"
        s = format_time(sub_data['start'])
        e = format_time(sub_data['end'] + (1.0 / self.fps_scan))
        self.update_signal.emit(s, e, sub_data['text'])

    def stop(self):
        self.is_running = False
        if self.capture_thread: self.capture_thread.stop()

# --- GUI CLASS ---
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.roi_rect_video = QRect(0, 0, 0, 0)
        self.is_drawing = False
        self.start_pos = None
        self.current_pos = None
        self.setMouseTracking(True)
        self.setScaledContents(False)

    def set_video_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.repaint()

    def get_scaling_info(self):
        if not self.original_pixmap: return 1.0, 0, 0
        w_widget = self.width()
        h_widget = self.height()
        w_img = self.original_pixmap.width()
        h_img = self.original_pixmap.height()
        if w_img == 0 or h_img == 0: return 1.0, 0, 0
        scale = min(w_widget / w_img, h_widget / h_img)
        w_scaled = int(w_img * scale)
        h_scaled = int(h_img * scale)
        x_offset = (w_widget - w_scaled) // 2
        y_offset = (h_widget - h_scaled) // 2
        return scale, x_offset, y_offset

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.original_pixmap:
            self.is_drawing = True
            self.start_pos = event.pos()
            self.current_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.current_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.current_pos = event.pos()
            scale, x_off, y_off = self.get_scaling_info()
            if scale == 0: return
            x1 = int((self.start_pos.x() - x_off) / scale)
            y1 = int((self.start_pos.y() - y_off) / scale)
            x2 = int((self.current_pos.x() - x_off) / scale)
            y2 = int((self.current_pos.y() - y_off) / scale)
            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x1 - x2)
            rh = abs(y1 - y2)
            rx = max(0, rx)
            ry = max(0, ry)
            if rx + rw > self.original_pixmap.width(): rw = self.original_pixmap.width() - rx
            if ry + rh > self.original_pixmap.height(): rh = self.original_pixmap.height() - ry
            self.roi_rect_video = QRect(rx, ry, rw, rh)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#000000"))
        if not self.original_pixmap:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chưa tải Video")
            return
        scale, x_off, y_off = self.get_scaling_info()
        w_scaled = int(self.original_pixmap.width() * scale)
        h_scaled = int(self.original_pixmap.height() * scale)
        target_rect = QRect(x_off, y_off, w_scaled, h_scaled)
        painter.drawPixmap(target_rect, self.original_pixmap)
        
        if not self.roi_rect_video.isEmpty() and not self.is_drawing:
            sx = int(self.roi_rect_video.x() * scale) + x_off
            sy = int(self.roi_rect_video.y() * scale) + y_off
            sw = int(self.roi_rect_video.width() * scale)
            sh = int(self.roi_rect_video.height() * scale)
            pen = QPen(QColor("#00ff00"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(sx, sy, sw, sh)

        if self.is_drawing and self.start_pos and self.current_pos:
            pen = QPen(QColor("#ff0000"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect_draw = QRect(self.start_pos, self.current_pos).normalized()
            painter.drawRect(rect_draw)

# --- MAIN APP ---
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Sub Extractor Ultimate (Review & Soi Mode)")
        self.setGeometry(100, 100, 1280, 850)
        
        self.video_queue = [] 
        self.current_video_idx = 0
        self.is_processing_batch = False
        self.is_single_mode = True 
        self.duration = 0
        self.roi_cache = (0,0,0,0)
        self.worker = None
        
        self.setStyleSheet(MODERN_STYLESHEET)
        setup_temp_folder()
        self.init_ui()

    def closeEvent(self, event):
        self.stop_ocr()
        cleanup_temp_folder()
        event.accept()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # HEADER
        top_frame = QFrame()
        top = QHBoxLayout(top_frame)
        top.setContentsMargins(0,0,0,0)
        
        self.btn_load = QPushButton("📂 CHỌN VIDEO")
        self.btn_load.setObjectName("btnLoad")
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.clicked.connect(self.load_videos)
        
        self.chk_live = QCheckBox("🔴 Live Monitor")
        self.chk_live.setChecked(True)
        
        self.chk_autoscroll = QCheckBox("📜 Cuộn theo kết quả")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setToolTip("Bỏ tích để dừng màn hình khi 'soi' lại kết quả")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["⚡ CPU Mode", "🚀 GPU (CUDA)"])
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🇨🇳 Trung (chi_sim)", "🇻🇳 Việt (vie)", "🇺🇸 Anh (eng)"])
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["Tốc độ: 5 fps", "Tốc độ: 10 fps", "Tốc độ: 20 fps", "Tốc độ: 25 fps"])
        self.fps_combo.setCurrentIndex(2) 

        top.addWidget(self.btn_load)
        top.addWidget(self.chk_live)
        top.addWidget(self.chk_autoscroll)
        top.addStretch()
        top.addWidget(QLabel("Cấu hình:"))
        top.addWidget(self.device_combo)
        top.addWidget(self.lang_combo)
        top.addWidget(self.fps_combo)
        layout.addWidget(top_frame)

        # BODY
        mid = QHBoxLayout()
        # Preview Area
        grp_vid = QGroupBox("1. MÀN HÌNH CHÍNH (Vẽ khung để chọn sub)")
        v_lo = QVBoxLayout()
        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 270)
        self.video_label.setStyleSheet("border: 1px solid #444;") 
        v_lo.addWidget(self.video_label, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000) 
        self.slider.sliderReleased.connect(self.seek_lazy)
        self.slider.setEnabled(False)
        v_lo.addWidget(self.slider)
        grp_vid.setLayout(v_lo)
        mid.addWidget(grp_vid, 50)

        # Live Monitor Area & Result
        right_layout = QVBoxLayout()
        
        # LIVE MONITOR BOX
        grp_live = QGroupBox("2. LIVE MONITOR (Soi chi tiết)")
        live_lo = QHBoxLayout()
        
        self.lbl_live_crop = QLabel("No Signal")
        self.lbl_live_crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_crop.setStyleSheet("background-color: #000; border: 1px dashed #555; min-width: 200px;")
        self.lbl_live_crop.setMinimumHeight(60)
        
        self.lbl_live_text = QLabel("...")
        self.lbl_live_text.setObjectName("lblLiveText")
        self.lbl_live_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_live_text.setWordWrap(True)
        
        live_lo.addWidget(self.lbl_live_crop, 40)
        live_lo.addWidget(self.lbl_live_text, 60)
        grp_live.setLayout(live_lo)
        right_layout.addWidget(grp_live)

        # RESULT TABLE
        grp_tbl = QGroupBox("3. KẾT QUẢ (Click vào dòng để soi)")
        t_lo = QVBoxLayout()
        self.lbl_batch_info = QLabel("Chế độ: Chưa chọn video")
        self.lbl_batch_info.setStyleSheet("color: #00ff7f; font-weight: bold;")
        t_lo.addWidget(self.lbl_batch_info)
        self.table = QTableWidget()
        self.table.setColumnCount(4) 
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Nội dung"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # KẾT NỐI SỰ KIỆN CLICK BẢNG
        self.table.cellClicked.connect(self.on_table_row_clicked)
        
        t_lo.addWidget(self.table)
        grp_tbl.setLayout(t_lo)
        right_layout.addWidget(grp_tbl)

        mid.addLayout(right_layout, 50)
        layout.addLayout(mid)

        # FOOTER
        btm_frame = QFrame()
        btm = QHBoxLayout(btm_frame)
        btm.setContentsMargins(0, 10, 0, 0)
        self.btn_start = QPushButton("▶ BẮT ĐẦU")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_process)
        self.btn_start.setEnabled(False)
        
        self.btn_stop = QPushButton("⏹ DỪNG LẠI")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_ocr)
        self.btn_stop.setEnabled(False)
        
        self.btn_save = QPushButton("💾 LƯU FILE SRT")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.manual_save)
        self.btn_save.setEnabled(False)

        self.lbl_status = QLabel("Sẵn sàng...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #00d2ff;")
        
        btm.addWidget(self.lbl_status)
        btm.addStretch()
        btm.addWidget(self.btn_start)
        btm.addWidget(self.btn_stop)
        btm.addWidget(self.btn_save) 
        layout.addWidget(btm_frame)

    def load_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Chọn Video (1 hoặc nhiều)", "", "Video Files (*.mp4 *.mkv *.avi *.ts *.mov)")
        if not paths: return
        self.video_queue = [os.path.abspath(p) for p in paths]
        self.current_video_idx = 0
        
        if len(self.video_queue) == 1:
            self.is_single_mode = True
            self.lbl_batch_info.setText(f"Chế độ: SINGLE - {os.path.basename(self.video_queue[0])}")
            self.btn_start.setText("▶ BẮT ĐẦU QUÉT")
        else:
            self.is_single_mode = False
            self.lbl_batch_info.setText(f"Chế độ: BATCH - {len(self.video_queue)} files")
            self.btn_start.setText(f"▶ CHẠY BATCH ({len(self.video_queue)} FILES)")
        
        self.load_preview(self.video_queue[0])
        self.btn_start.setEnabled(True)
        self.btn_save.setEnabled(False)

    def load_preview(self, path):
        w, h, duration = get_video_info(path)
        if duration == 0:
            QMessageBox.critical(self, "Lỗi", f"Không đọc được file: {os.path.basename(path)}")
            return
        self.duration = duration
        self.slider.setEnabled(True)
        self.slider.setValue(0)
        self.show_frame_at(0, path) 
        self.lbl_status.setText(f"✅ Đang xem: {os.path.basename(path)}")

    def seek_lazy(self):
        if not self.video_queue: return
        val = self.slider.value()
        seconds = (val / 1000) * self.duration
        self.show_frame_at(seconds, self.video_queue[self.current_video_idx])

    def show_frame_at(self, seconds, path):
        img, err = get_preview_frame(path, seconds)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.video_label.set_video_pixmap(pixmap)
            return pixmap # Trả về pixmap để dùng cho crop
        else:
            self.lbl_status.setText("❌ Lỗi preview ảnh!")
            return None

    # --- CHỨC NĂNG SOI KẾT QUẢ (QUAN TRỌNG) ---
    def on_table_row_clicked(self, row, col):
        # Lấy thông tin dòng
        start_str = self.table.item(row, 1).text()
        text_content = self.table.item(row, 3).text()
        
        # Parse thời gian
        seconds = self.parse_timestamp(start_str)
        current_vid = self.video_queue[self.current_video_idx]
        
        # Cập nhật Màn hình chính
        full_pixmap = self.show_frame_at(seconds, current_vid)
        
        # Cập nhật Màn hình Live Monitor (Crop)
        if full_pixmap and self.roi_cache != (0,0,0,0):
            x, y, w, h = self.roi_cache
            # Đảm bảo ROI hợp lệ với ảnh
            if x+w <= full_pixmap.width() and y+h <= full_pixmap.height():
                cropped = full_pixmap.copy(QRect(x, y, w, h))
                scaled_crop = cropped.scaled(self.lbl_live_crop.size(), Qt.AspectRatioMode.KeepAspectRatio)
                self.lbl_live_crop.setPixmap(scaled_crop)
        
        # Cập nhật chữ
        self.lbl_live_text.setText(text_content)
        self.lbl_status.setText(f"🔎 Đang soi: {start_str}")

    def start_process(self):
        roi_rect = self.video_label.roi_rect_video
        self.roi_cache = (roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height())

        if self.roi_cache == (0,0,0,0) or self.roi_cache[2] < 5 or self.roi_cache[3] < 5:
             QMessageBox.warning(self, "Chưa chọn vùng", "Hãy vẽ khung chữ trên Video Preview trước!")
             return

        self.is_processing_batch = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.slider.setEnabled(False) 
        
        self.run_current_video()

    def run_current_video(self):
        if self.current_video_idx >= len(self.video_queue):
            self.on_all_complete()
            return

        video_path = self.video_queue[self.current_video_idx]
        vid_name = os.path.basename(video_path)
        
        if self.is_single_mode:
            self.lbl_batch_info.setText(f"Đang xử lý: {vid_name}")
        else:
            self.lbl_batch_info.setText(f"🔄 Batch: {self.current_video_idx + 1}/{len(self.video_queue)} - {vid_name}")
            
        self.table.setRowCount(0) 

        use_gpu = (self.device_combo.currentIndex() == 1)
        lang_map = {0: 'ch', 1: 'vi', 2: 'en'}
        lang = lang_map.get(self.lang_combo.currentIndex(), 'ch')
        fps_text = self.fps_combo.currentText()
        fps_scan = 5
        if "10 fps" in fps_text: fps_scan = 10
        elif "20 fps" in fps_text: fps_scan = 20
        elif "25 fps" in fps_text: fps_scan = 25
        
        live_mode = self.chk_live.isChecked()

        self.worker = OCRWorker(video_path, self.roi_cache, use_gpu, lang, fps_scan, live_mode)
        self.worker.update_signal.connect(self.add_row)
        self.worker.live_signal.connect(self.update_live_monitor)
        self.worker.progress_signal.connect(lambda v: self.lbl_status.setText(f"🚀 {vid_name}: {v}%"))
        self.worker.error_signal.connect(lambda e: self.lbl_status.setText(f"⚠️ Lỗi: {e}"))
        self.worker.log_signal.connect(lambda s: self.lbl_status.setText(s))
        self.worker.finished_signal.connect(self.on_video_finished)
        self.worker.start()
        
    def update_live_monitor(self, qt_img, text):
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.lbl_live_crop.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.lbl_live_crop.setPixmap(scaled)
        if text: self.lbl_live_text.setText(text)
        else: self.lbl_live_text.setText("...")

    def stop_ocr(self):
        self.is_processing_batch = False
        if self.worker: self.worker.stop()
        self.lbl_status.setText("🛑 Đã dừng.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load.setEnabled(True)
        self.slider.setEnabled(True) 
        if self.is_single_mode and self.table.rowCount() > 0:
            self.btn_save.setEnabled(True)

    def on_video_finished(self):
        if not self.is_processing_batch: return 
        current_vid = self.video_queue[self.current_video_idx]
        if self.is_single_mode:
            self.lbl_status.setText("✅ Đã xong! Hãy bấm Lưu SRT.")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_load.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.slider.setEnabled(True) 
            QMessageBox.information(self, "Hoàn tất", "Đã quét xong video!\nHãy kiểm tra bảng và bấm 'LƯU FILE SRT'.")
        else:
            self.auto_save_srt(current_vid)
            self.current_video_idx += 1
            self.run_current_video()

    def on_all_complete(self):
        self.lbl_batch_info.setText("✅ ĐÃ HOÀN THÀNH BATCH!")
        self.lbl_status.setText("Batch hoàn tất.")
        QMessageBox.information(self, "Batch Complete", f"Đã xử lý xong {len(self.video_queue)} video!\nFile sub đã lưu tại thư mục gốc.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_load.setEnabled(True)
        self.slider.setEnabled(True)
        self.is_processing_batch = False

    def add_row(self, s, e, t):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
        self.table.setItem(r, 1, QTableWidgetItem(s))
        self.table.setItem(r, 2, QTableWidgetItem(e))
        self.table.setItem(r, 3, QTableWidgetItem(t))
        
        # LOGIC AUTO SCROLL
        if self.chk_autoscroll.isChecked():
            self.table.scrollToBottom()

    def parse_timestamp(self, t_str):
        try:
            time_part, ms_part = t_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            return h * 3600 + m * 60 + s + int(ms_part) / 1000.0
        except: return 0.0

    def format_timestamp(self, seconds):
        total_sec = int(seconds)
        ms = int((seconds - total_sec) * 1000)
        base = str(datetime.timedelta(seconds=total_sec))
        if len(base.split(":")) < 3: base = "0" + base 
        return f"{base},{ms:03d}"

    def optimize_timeline(self, raw_subs, chars_per_sec=4.5, min_gap=0.1):
        if not raw_subs: return []
        subs = sorted([s for s in raw_subs if s['text'].strip()], key=lambda k: k['start'])
        processed = []
        if not subs: return []

        for i in range(len(subs)):
            current = subs[i].copy()
            char_count = len(current['text'].replace(" ", ""))
            ideal_duration = char_count / chars_per_sec
            if ideal_duration < 1.0: ideal_duration = 1.0 

            if i < len(subs) - 1:
                next_start = subs[i+1]['start']
                max_end = next_start - min_gap
            else:
                max_end = current['start'] + ideal_duration + 2.0

            if processed:
                prev_end = processed[-1]['end']
                gap = current['start'] - prev_end
                if 0 < gap < min_gap:
                    current['start'] = prev_end 

            current_duration = current['end'] - current['start']
            if current_duration < ideal_duration:
                new_end = current['start'] + ideal_duration
                if new_end > max_end:
                    current['end'] = max_end
                else:
                    current['end'] = new_end
            
            if current['end'] <= current['start']:
                 current['end'] = current['start'] + 0.5
            processed.append(current)
        return processed

    def get_subs_from_table(self):
        raw_subs = []
        rows = self.table.rowCount()
        for r in range(rows):
            s_str = self.table.item(r, 1).text()
            e_str = self.table.item(r, 2).text()
            text = self.table.item(r, 3).text()
            raw_subs.append({'start': self.parse_timestamp(s_str), 'end': self.parse_timestamp(e_str), 'text': text})
        return raw_subs

    def manual_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu SRT", "", "Subtitle (*.srt)")
        if not path: return
        raw_subs = self.get_subs_from_table()
        if not raw_subs: return
        final_subs = self.optimize_timeline(raw_subs, chars_per_sec=4.5, min_gap=0.1)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for i, sub in enumerate(final_subs):
                    s_fmt = self.format_timestamp(sub['start'])
                    e_fmt = self.format_timestamp(sub['end'])
                    f.write(f"{i+1}\n{s_fmt} --> {e_fmt}\n{sub['text']}\n\n")
            QMessageBox.information(self, "Thành công", f"Đã lưu file: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def auto_save_srt(self, video_path):
        base_name = os.path.splitext(video_path)[0]
        srt_path = base_name + ".srt"
        raw_subs = self.get_subs_from_table()
        if not raw_subs: return
        final_subs = self.optimize_timeline(raw_subs, chars_per_sec=4.5, min_gap=0.1)
        try:
            with open(srt_path, 'w', encoding='utf-8') as f:
                for i, sub in enumerate(final_subs):
                    s_fmt = self.format_timestamp(sub['start'])
                    e_fmt = self.format_timestamp(sub['end'])
                    f.write(f"{i+1}\n{s_fmt} --> {e_fmt}\n{sub['text']}\n\n")
            print(f"Auto-saved: {srt_path}")
        except Exception as e:
            print(f"Lỗi lưu file {srt_path}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
