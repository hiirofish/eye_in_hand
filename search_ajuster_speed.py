#!/usr/bin/env python3
"""
==============================================================================
Dobot Eye-in-Hand 物体探索 v7.8 (完全統合・インテリジェント版)
==============================================================================
"""

import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
import time
import threading
from datetime import datetime

try:
    import DobotDllType as dType
    DOBOT_AVAILABLE = True
except ImportError:
    DOBOT_AVAILABLE = False

# ======================== BlockDetector (リアルタイム調整対応) ========================
class BlockDetector:
    def __init__(self):
        # ユーザーが見つけた最適な初期値
        self.h_min = 0;   self.h_max = 178
        self.s_min = 0;   self.s_max = 56
        self.v_min = 179;  self.v_max = 255
        self.min_area = 500
        self.min_sol = 0.70
        self.roi_top = 0.35; self.roi_bot = 0.98
        self.margin_x = 0.08
        self.kern = np.ones((5, 5), np.uint8)
        self.last_mask = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        rt, rb = int(h * self.roi_top), int(h * self.roi_bot)
        roi = frame[rt:rb, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([self.h_min, self.s_min, self.v_min]), 
                               np.array([self.h_max, self.s_max, self.v_max]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kern)
        mask = cv2.dilate(mask, self.kern, iterations=1)
        self.last_mask = np.zeros((h, w), dtype=np.uint8)
        self.last_mask[rt:rb, :] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area: continue
            sol = a / cv2.contourArea(cv2.convexHull(c)) if cv2.contourArea(cv2.convexHull(c)) > 0 else 0
            if sol > self.min_sol: valid.append((c, a, cv2.minAreaRect(c)))
        if not valid: return None
        _, _, r = max(valid, key=lambda v: v[1])
        co = cv2.boxPoints(r); co[:, 1] += rt
        return dict(center=(int(r[0][0]), int(r[0][1] + rt)), area=cv2.contourArea(cv2.convexHull(np.int32(co))),
                    box=np.int32(co))

# ======================== メインGUI ========================
class ObjectSearchApp:
    CALIB_Z1, CALIB_X1, CALIB_Y1 = -31.9, 313, 223
    CALIB_Z2, CALIB_X2, CALIB_Y2 = 88.0, 276, 403

    def __init__(self, root):
        self.root = root
        self.root.title("物体探索 v7.8 (Intelligent Full Control)")

        # --- 内部パラメータ ---
        self.SERVO_MAX_ITER = 60
        self.SERVO_GAIN = 0.0015
        self.CONVERGE_PX = 15
        self.MIN_JOG_TIME = 0.05
        self.MAX_JOG_TIME = 0.30
        self.DECAY_FACTORS = [1.0, 0.75, 0.50] # 3段階減衰

        self.api = dType.load() if DOBOT_AVAILABLE else None
        self.is_connected = False
        self.is_camera_running = False
        self.searching = False
        self.view_mask = tk.BooleanVar(value=False)
        self.stop_flag = threading.Event()
        self.frame_lock = threading.Lock()
        
        self.detector = BlockDetector()
        self.current_det = None
        self.current_z_for_calc = 0.0

        self.setup_ui()
        self.root.after(500, self._periodic_update)

    def setup_ui(self):
        main = tk.Frame(self.root); main.pack(padx=5, pady=5, fill="both", expand=True)
        left = tk.Frame(main); left.pack(side="left", padx=5, fill="y")

        # 1. ロボット接続 & 手動操作 (復活)
        f1 = tk.LabelFrame(left, text="1. Robot & Manual", font=("Arial", 9, "bold")); f1.pack(pady=2, fill="x")
        tk.Button(f1, text="Connect", command=self.connect, bg="lightblue", width=12).pack(pady=3)
        self.conn_label = tk.Label(f1, text="未接続"); self.conn_label.pack()
        
        g = tk.Frame(f1); g.pack(pady=2)
        self._jog_btn(g, "J1左", 0, 0, "j1", 1, "#e0e0ff")
        self._jog_btn(g, "J1右", 0, 1, "j1", -1, "#e0e0ff")
        self._jog_btn(g, "伸長", 0, 2, "reach", 1, "#e0ffe0")
        self._jog_btn(g, "短縮", 0, 3, "reach", -1, "#e0ffe0")
        
        g2 = tk.Frame(f1); g2.pack(pady=2)
        self._jog_btn_raw(g2, "↑上", 0, 0, 5)
        self._jog_btn_raw(g2, "↓下", 0, 1, 6)
        tk.Button(g2, text="開く", bg="green", fg="white", command=lambda: self.gripper_action(False)).grid(row=0, column=2, padx=2)
        tk.Button(g2, text="閉じる", bg="red", fg="white", command=lambda: self.gripper_action(True)).grid(row=0, column=3, padx=2)

        # 2. HSV調整パネル (復活)
        self.setup_hsv_ui(left)

        # 3. スピード・サーボ設定 (復活)
        self.setup_servo_ui(left)

        # 4. 探索ボタン
        self.search_btn = tk.Button(left, text="▶ インテリジェント探索開始", command=self.toggle_search, 
                                   bg="#2266aa", fg="white", font=("Arial", 11, "bold"), height=2)
        self.search_btn.pack(pady=10, fill="x")
        tk.Checkbutton(left, text="マスク画像を表示", variable=self.view_mask).pack()

        # 右側：映像 & ログ
        right = tk.Frame(main); right.pack(side="left", padx=5, fill="both", expand=True)
        btn_f = tk.Frame(right); btn_f.pack(fill="x")
        tk.Button(btn_f, text="カメラ開始", command=self.start_camera, bg="lightgreen", width=12).pack(side="left", padx=5)
        tk.Button(btn_f, text="カメラ停止", command=self.stop_camera, width=12).pack(side="left")

        self.camera_label = tk.Label(right, bg="black", width=640, height=480); self.camera_label.pack(pady=5)
        self.log_text = scrolledtext.ScrolledText(right, height=12, font=("Consolas", 8)); self.log_text.pack(fill="both")

    def setup_hsv_ui(self, parent):
        f = tk.LabelFrame(parent, text="2. HSV Adjustment", font=("Arial", 9, "bold")); f.pack(fill="x", pady=2)
        def add_s(label, attr, mx):
            fr = tk.Frame(f); fr.pack(fill="x")
            tk.Label(fr, text=label, width=6).pack(side="left")
            s = tk.Scale(fr, from_=0, to=mx, orient="horizontal", command=lambda v: setattr(self.detector, attr, int(v)))
            s.set(getattr(self.detector, attr)); s.pack(side="right", fill="x", expand=True)
        add_s("H_Max", "h_max", 180)
        add_s("S_Max", "s_max", 255)
        add_s("V_Min", "v_min", 255)

    def setup_servo_ui(self, parent):
        f = tk.LabelFrame(parent, text="3. Speed Tuning", font=("Arial", 9, "bold"), fg="darkgreen"); f.pack(fill="x", pady=2)
        fr = tk.Frame(f); fr.pack(fill="x")
        tk.Label(fr, text="感度G", width=6).pack(side="left")
        s = tk.Scale(fr, from_=5, to=50, orient="horizontal", label="x0.0001",
                     command=lambda v: setattr(self, 'SERVO_GAIN', int(v) * 0.0001))
        s.set(int(self.SERVO_GAIN / 0.0001)); s.pack(side="right", fill="x", expand=True)
        fr2 = tk.Frame(f); fr2.pack(fill="x")
        tk.Label(fr2, text="収束PX", width=6).pack(side="left")
        s2 = tk.Scale(fr2, from_=5, to=40, orient="horizontal", command=lambda v: setattr(self, 'CONVERGE_PX', int(v)))
        s2.set(self.CONVERGE_PX); s2.pack(side="right", fill="x", expand=True)

    # --- JOG移動系 ---
    def _jog_btn(self, p, t, r, c, m, d, clr):
        b = tk.Button(p, text=t, width=6, bg=clr); b.grid(row=r, column=c, padx=2)
        b.bind("<ButtonPress-1>", lambda e: dType.SetJOGCmd(self.api, m=="j1", 1 if d>0 else 2, isQueued=1))
        b.bind("<ButtonRelease-1>", lambda e: dType.SetJOGCmd(self.api, False, 0, isQueued=1))

    def _jog_btn_raw(self, p, t, r, c, cid):
        b = tk.Button(p, text=t, width=6); b.grid(row=r, column=c, padx=2)
        b.bind("<ButtonPress-1>", lambda e: dType.SetJOGCmd(self.api, False, cid, isQueued=1))
        b.bind("<ButtonRelease-1>", lambda e: dType.SetJOGCmd(self.api, False, 0, isQueued=1))

    def gripper_action(self, close):
        if not self.is_connected: return
        dType.SetQueuedCmdClear(self.api)
        if close:
            dType.SetEndEffectorSuctionCupEx(self.api, True, True, isQueued=1)
            dType.SetEndEffectorGripperEx(self.api, True, True, isQueued=1)
        else:
            dType.SetEndEffectorSuctionCupEx(self.api, True, True, isQueued=1)
            dType.SetEndEffectorGripperEx(self.api, True, False, isQueued=1)
            dType.SetQueuedCmdStartExec(self.api); time.sleep(0.5)
            dType.SetEndEffectorSuctionCupEx(self.api, False, False, isQueued=1)
            dType.SetEndEffectorGripperEx(self.api, False, False, isQueued=1)
        dType.SetQueuedCmdStartExec(self.api)

    # --- インテリジェント・サーボロジック (3段階減衰 + 交互制御) ---
    def _jog_servo_phase1(self):
        self.log("===== Intelligent Servo Sequence Start =====")
        discovery_count = 0
        
        for i in range(self.SERVO_MAX_ITER):
            if self.stop_flag.is_set(): return False
            time.sleep(0.5) # 静止待ち
            
            with self.frame_lock: det = self.current_det
            if not det: 
                discovery_count = 0; self.log(f"iter {i}: Object Lost."); continue
            
            discovery_count += 1
            if discovery_count < 2: # 2回連続発見で確定
                self.log(f"iter {i}: Confirming Discovery... ({discovery_count}/2)"); continue

            # ズレ(dx, dy)の算出
            cx, cy = det['center']
            z = self.current_z_for_calc
            ratio = (z - self.CALIB_Z1) / (self.CALIB_Z2 - self.CALIB_Z1)
            tx = int(self.CALIB_X1 + (self.CALIB_X2 - self.CALIB_X1) * ratio)
            ty = int(self.CALIB_Y1 + (self.CALIB_Y2 - self.CALIB_Y1) * ratio)
            dx, dy = cx - tx, cy - ty

            if abs(dx) < self.CONVERGE_PX and abs(dy) < self.CONVERGE_PX:
                self.log(f"★ 収束成功! Error: dx={dx}, dy={dy}"); return True

            # 減衰率(Factor)の決定
            if i < len(self.DECAY_FACTORS):
                mode = "PHASE A: COARSE"
                factor = self.DECAY_FACTORS[i] # 1.0 -> 0.75 -> 0.5
            else:
                mode = "PHASE B: FINE"
                factor = 0.30 

            # 移動時間の計算ログ出力
            move_x = max(self.MIN_JOG_TIME, min(self.MAX_JOG_TIME, abs(dx) * self.SERVO_GAIN * factor))
            move_y = max(self.MIN_JOG_TIME, min(self.MAX_JOG_TIME, abs(dy) * self.SERVO_GAIN * factor))
            
            self.log(f"[{mode}] Iter:{i} Factor:{factor:.2f}")
            self.log(f"  > dx:{dx:+.0f} dy:{dy:+.0f} | Calc_X:{move_x:.3f}s Calc_Y:{move_y:.3f}s")

            # X軸 -> Y軸 交互制御
            if abs(dx) >= self.CONVERGE_PX:
                dType.SetJOGCmd(self.api, True, (2 if dx > 0 else 1), isQueued=1)
                time.sleep(move_x); dType.SetJOGCmd(self.api, False, 0, isQueued=1); time.sleep(0.1)

            if abs(dy) >= self.CONVERGE_PX:
                dType.SetJOGCmd(self.api, False, (2 if dy > 0 else 1), isQueued=1)
                time.sleep(move_y); dType.SetJOGCmd(self.api, False, 0, isQueued=1)
        return False

    # --- カメラ & 接続系 ---
    def connect(self):
        if DOBOT_AVAILABLE and dType.ConnectDobot(self.api, "COM3", 115200)[0] == 0:
            self.is_connected = True
            dType.SetQueuedCmdStartExec(self.api)
            dType.SetJOGCommonParams(self.api, 60, 60, isQueued=1)
            self.conn_label.config(text="接続済 ✓", fg="green")
            self.log("Dobot Connected")

    def toggle_search(self):
        if self.searching: self.stop_flag.set(); self.searching = False
        else:
            self.searching = True; self.stop_flag.clear()
            self.search_btn.config(text="■ 停止", bg="red")
            threading.Thread(target=lambda: (self._jog_servo_phase1(), setattr(self, 'searching', False), 
                                           self.search_btn.config(text="▶ インテリジェント探索開始", bg="#2266aa")), daemon=True).start()

    def start_camera(self):
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.is_camera_running = True; self._update_camera()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap: self.cap.release()

    def _update_camera(self):
        if not self.is_camera_running: return
        ret, frame = self.cap.read()
        if ret:
            det = self.detector.detect(frame)
            with self.frame_lock: self.current_det = det
            display = cv2.cvtColor(self.detector.last_mask, cv2.COLOR_GRAY2BGR) if self.view_mask.get() else frame.copy()
            # ターゲット位置算出と描画
            z = self.current_z_for_calc
            ratio = (z - self.CALIB_Z1) / (self.CALIB_Z2 - self.CALIB_Z1)
            tx = int(self.CALIB_X1 + (self.CALIB_X2 - self.CALIB_X1) * ratio)
            ty = int(self.CALIB_Y1 + (self.CALIB_Y2 - self.CALIB_Y1) * ratio)
            cv2.drawMarker(display, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
            if det:
                cv2.drawContours(display, [det['box']], 0, (0, 255, 0), 2)
                cv2.arrowedLine(display, (tx, ty), det['center'], (255, 255, 0), 2)
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)))
            self.camera_label.imgtk = img; self.camera_label.config(image=img)
        self.root.after(33, self._update_camera)

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n"); self.log_text.see(tk.END); print(msg)

    def _periodic_update(self):
        if self.is_connected: self.current_z_for_calc = dType.GetPose(self.api)[2]
        self.root.after(500, self._periodic_update)

if __name__ == "__main__":
    root = tk.Tk(); app = ObjectSearchApp(root); root.mainloop()