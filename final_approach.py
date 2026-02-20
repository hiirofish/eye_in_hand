#!/usr/bin/env python3
"""
==============================================================================
Dobot Eye-in-Hand v10.2 (床基準Z座標・TGT JSON対応)
==============================================================================
修正履歴:
 v10.0: キャリブレーション統合・サーボ降下最適化
 v10.1: TGTキャリブレーション値を tgt_calibration.json から読み込み
 v10.2:
   ★ 全Z値を「床からの相対値」に変換
     → Step 0 でハンドを床につけた時のZ値を floor_z として記録
     → Z_GRASP, Z_APPROACH, Z_DESCENT, Z_LIFT 全て床基準
     → TGTキャリブのZ値も floor_z でオフセット補正
     → 電源再投入でZ原点がずれても Step 0 をやり直すだけで復帰
   
   Step 0 手順: ハンドを開き、カメラと水平にし、床につけた状態で保存
==============================================================================
"""

import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
import math
import time
import threading
from datetime import datetime
import sys

# ★ eye_in_handサブフォルダから親ディレクトリのDobotDllType.py/DobotDll.dllを参照
_parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _parent_dir)
os.chdir(_parent_dir)

try:
    import DobotDllType as dType
    DOBOT_AVAILABLE = True
except ImportError:
    DOBOT_AVAILABLE = False
    print("[WARN] DobotDllType not found")


# ======================== BlockDetector ========================
class BlockDetector:
    def __init__(self):
        self.h_min = 0;   self.h_max = 178
        self.s_min = 0;   self.s_max = 100
        self.v_min = 179;  self.v_max = 255
        self.min_area = 300
        self.max_area = 30000 
        self.min_aspect = 1.2
        self.max_aspect = 8.0 
        self.min_rect = 0.25   
        self.min_sol = 0.40
        self.roi_top = 0.35
        self.roi_bot = 0.98
        self.margin_x = 0.08
        self.kern = np.ones((5, 5), np.uint8)
        self.debug = []
        self.last_mask = None

    @property
    def lower(self):
        return np.array([self.h_min, self.s_min, self.v_min])
    @property
    def upper(self):
        return np.array([self.h_max, self.s_max, self.v_max])

    def detect(self, frame):
        h, w = frame.shape[:2]
        rt, rb = int(h * self.roi_top), int(h * self.roi_bot)
        ml, mr = int(w * self.margin_x), int(w * (1 - self.margin_x))
        roi = frame[rt:rb, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kern)
        mask = cv2.dilate(mask, self.kern, iterations=1)
        self.last_mask = np.zeros((h, w), dtype=np.uint8)
        self.last_mask[rt:rb, :] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        self.debug = []
        valid = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < 100: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + rt
            info = dict(cx=cx, cy=cy, area=a, ok=False, why="")
            if a < self.min_area:       info['why'] = f'Small({a:.0f})'
            elif a > self.max_area:     info['why'] = f'Large({a:.0f})'
            else:
                r = cv2.minAreaRect(c)
                ww, hh = r[1]
                if ww == 0 or hh == 0:  info['why'] = 'size0'
                else:
                    asp = max(ww, hh) / min(ww, hh)
                    rec = a / (ww * hh)
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    sol = a / hull_area if hull_area > 0 else 0
                    if   asp < self.min_aspect: info['why'] = f'Square({asp:.1f})'
                    elif asp > self.max_aspect: info['why'] = f'Slim({asp:.1f})'
                    elif rec < self.min_rect:   info['why'] = f'Rect({rec:.2f})'
                    elif sol < self.min_sol:    info['why'] = f'Sol({sol:.2f})'
                    elif cx < ml or cx > mr:    info['why'] = f'Edge(x={cx})'
                    else:
                        info['ok'] = True
                        valid.append((c, a, r, rec, sol))
            self.debug.append(info)
        if not valid: return None
        best_c, best_a, _, best_rec, best_sol = max(valid, key=lambda v: v[1])
        co = best_c.copy(); co[:, :, 1] += rt
        rect = cv2.minAreaRect(co)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        ww, hh = rect[1]; ang = rect[2]
        short = ang if ww < hh else ang + 90
        while short > 90:  short -= 180
        while short < -90: short += 180
        box = np.int32(cv2.boxPoints(rect))
        return dict(center=(cx, cy), area=best_a, short_angle=short,
                    box=box, contour=co, rect=best_rec, sol=best_sol)

    def draw(self, frame, det):
        for d in self.debug:
            c = (0, 255, 0) if d['ok'] else (0, 0, 255)
            cv2.circle(frame, (d['cx'], d['cy']), 4, c, -1)
            if not d['ok'] and d.get('area', 0) >= self.min_area:
                cv2.putText(frame, d['why'][:15], (d['cx'] + 5, d['cy']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        if det:
            cx, cy = det['center']
            cv2.drawContours(frame, [det['box']], 0, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"Ang:{det['short_angle']:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"({cx},{cy}) A:{det['area']:.0f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            n = len(self.debug)
            ok = sum(1 for d in self.debug if d['ok'])
            cv2.putText(frame, f"No object (cand:{n} pass:{ok})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame


# ======================== メインGUI ========================
class ObjectSearchApp:

    # ★ v10.2: TGTデフォルト値（フォールバック）
    CALIB_Z1_DEFAULT = None
    CALIB_X1_DEFAULT = None
    CALIB_Y1_DEFAULT = None
    CALIB_Z2_DEFAULT = None
    CALIB_X2_DEFAULT = None
    CALIB_Y2_DEFAULT = None
    CALIB_FLOOR_Z_DEFAULT = None
    TARGET_X = 285 
    
    # ★ v10.2: 全Z値を「床からの相対値(mm)」で定義
    Z_GRASP_REL = 2.0        # 把持高さ: 床+2mm
    Z_APPROACH_REL = 32.0    # R軸合わせ時: 床+32mm
    Z_LIFT_REL = 52.0        # 持ち上げ後: 床+52mm
    Z_DESCENT_REL = [62.0, 42.0, 32.0]  # サーボ降下ステップ（床相対）
    
    BLIND_OFFSET_TIME = 0.45 
    COARSE_PX = 30
    CONVERGE_PX = 15
    CALIB_MOVE_TIME = 0.5
    CALIB_MIN_PX = 10
    SERVO_MAX_ITER = 40
    
    JOG_SCAN_TIME = 0.4
    JOG_SERVO_TIME = 0.10
    JOG_SERVO_LONG = 0.20
    SETTLE_TIME = 0.4
    
    GAIN_SCHEDULE = {1: 1.0, 2: 0.75}
    GAIN_DEFAULT = 0.5
    PROP_MAX_TIME = 0.5
    PROP_MIN_TIME = 0.05

    # R軸パラメータ
    R_MIN = -35.6
    R_MAX = 155.5

    def __init__(self, root):
        self.root = root
        self.root.title("物体探索 v10.2 (床基準Z)")

        self.api = dType.load() if DOBOT_AVAILABLE else None
        self.is_connected = False
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.current_det = None
        self.frame_lock = threading.Lock()
        self.detector = BlockDetector()
        self.searching = False
        self.stop_flag = threading.Event()
        self.current_z_for_calc = 0.0
        
        self.last_detected_angle = 0.0
        
        self.calib_j1_sec_per_px = None
        self.calib_reach_sec_per_px = None
        
        # ★ v10.2: 床Z基準値（Step 0で記録）
        self.floor_z = None
        
        # ★ v10.2: TGTキャリブ値
        self.CALIB_Z1 = self.CALIB_Z1_DEFAULT
        self.CALIB_X1 = self.CALIB_X1_DEFAULT
        self.CALIB_Y1 = self.CALIB_Y1_DEFAULT
        self.CALIB_Z2 = self.CALIB_Z2_DEFAULT
        self.CALIB_X2 = self.CALIB_X2_DEFAULT
        self.CALIB_Y2 = self.CALIB_Y2_DEFAULT
        self.calib_floor_z = self.CALIB_FLOOR_Z_DEFAULT
        
        # ★ v10.0: 統合キャリブレーション基準値
        self.hand_base_diff = None

        self.r_offset = 106.9
        self.scan_j1_min = None
        self.scan_j1_max = None
        self.last_move = None
        self._load_configs()

        self.setup_ui()
        print("=" * 70)
        print("  物体探索 v10.2 (床基準Z座標)")
        print("=" * 70)
        if self.floor_z is not None:
            print(f"  床Z: {self.floor_z:.1f}mm")
        else:
            print(f"  床Z: 未設定 ⚠ (Step 0でキャリブしてください)")
        if self.hand_base_diff is not None:
            print(f"  ハンド基準(R-J1): {self.hand_base_diff:.1f}°")
        else:
            print(f"  ハンド基準: 未設定 ⚠")
        print(f"  Z相対値: GRASP=+{self.Z_GRASP_REL} APPROACH=+{self.Z_APPROACH_REL} LIFT=+{self.Z_LIFT_REL}")
        print(f"  降下ステップ(床相対): {self.Z_DESCENT_REL}")
        print(f"  TGT点1: Z={self.CALIB_Z1} → px=({self.CALIB_X1}, {self.CALIB_Y1})")
        print(f"  TGT点2: Z={self.CALIB_Z2} → px=({self.CALIB_X2}, {self.CALIB_Y2})")
        print("=" * 70)

    def _fp(self, name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    # ★ v10.2: Z座標変換ヘルパー
    def _abs_z(self, rel_z):
        """床からの相対Z → Dobot絶対Z"""
        if self.floor_z is None:
            self.log("⚠ floor_z未設定！Step 0でキャリブしてください", "ERROR")
            return 0.0
        return self.floor_z + rel_z

    def _rel_z(self, abs_z):
        """Dobot絶対Z → 床からの相対Z"""
        if self.floor_z is None: return 0.0
        return abs_z - self.floor_z

    def _tgt_abs_z(self, raw_z):
        """TGTキャリブのZ値を現在のfloor_zでオフセット補正"""
        rel = raw_z - self.calib_floor_z
        if self.floor_z is None: return raw_z
        return self.floor_z + rel

    def _load_configs(self):
        # ★ TGT JSON対応: tgt_calibration.json 読み込み
        fp = self._fp("tgt_calibration.json")
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            self.CALIB_Z1 = d.get('calib_z1', self.CALIB_Z1_DEFAULT)
            self.CALIB_X1 = d.get('calib_x1', self.CALIB_X1_DEFAULT)
            self.CALIB_Y1 = d.get('calib_y1', self.CALIB_Y1_DEFAULT)
            self.CALIB_Z2 = d.get('calib_z2', self.CALIB_Z2_DEFAULT)
            self.CALIB_X2 = d.get('calib_x2', self.CALIB_X2_DEFAULT)
            self.CALIB_Y2 = d.get('calib_y2', self.CALIB_Y2_DEFAULT)
            self.calib_floor_z = d.get('floor_z', self.CALIB_FLOOR_Z_DEFAULT)
            saved_at = d.get('saved_at', '不明')
            print(f"[INFO] TGTキャリブ読み込み ({saved_at}, calib_floor_z={self.calib_floor_z})")
        else:
            print(f"[INFO] tgt_calibration.json なし → デフォルト値使用")

        # ★ v10.2: ハンドキャリブ + floor_z 読み込み
        fp = self._fp("hand_calibration.json")
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            self.hand_base_diff = d.get('hand_base_diff')
            self.floor_z = d.get('floor_z')
            print(f"[INFO] ハンド基準(R-J1)={self.hand_base_diff}°")
            if self.floor_z is not None:
                print(f"[INFO] 床Z={self.floor_z:.1f}mm")
            else:
                print(f"[INFO] 床Z=未記録 → 再キャリブ推奨")

        fp = self._fp("gripper_calibration.json")
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            self.r_offset = d.get('r_offset', self.r_offset)

        fp = self._fp("scan_area.json")
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            self.scan_j1_min = d.get('scan_j1_min')
            self.scan_j1_max = d.get('scan_j1_max')
            print(f"[INFO] scan_area: J1=[{self.scan_j1_min:.1f}, {self.scan_j1_max:.1f}]")

    def setup_ui(self):
        main = tk.Frame(self.root)
        main.pack(padx=5, pady=5, fill="both", expand=True)
        left = tk.Frame(main)
        left.pack(side="left", padx=5, fill="y")

        # 接続
        f = tk.LabelFrame(left, text="接続", font=("MS Gothic", 9, "bold"))
        f.pack(pady=3, fill="x")
        r = tk.Frame(f); r.pack(fill="x", padx=3, pady=3)
        tk.Button(r, text="Connect", command=self.connect, bg="lightblue", width=10).pack(side="left", padx=2)
        tk.Button(r, text="ALM Reset", command=self._clear_alarm, bg="#ff6666", fg="white", width=9).pack(side="left", padx=2)
        self.conn_label = tk.Label(f, text="未接続", fg="gray")
        self.conn_label.pack()

        # 現在位置
        f = tk.LabelFrame(left, text="現在位置", font=("MS Gothic", 9))
        f.pack(pady=3, fill="x")
        self.pos_label = tk.Label(f, text="---", font=("Consolas", 9), justify="left", anchor="w")
        self.pos_label.pack(padx=5, pady=3, fill="x")

        # 手動操作
        f = tk.LabelFrame(left, text="手動操作", font=("MS Gothic", 9, "bold"))
        f.pack(pady=3, fill="x")
        g = tk.Frame(f); g.pack(padx=5, pady=3)
        tk.Label(g, text="J1回転", font=("MS Gothic", 8), fg="gray").grid(row=0, column=0, columnspan=2)
        self._jog_btn(g, "← 左", 1, 0, "j1", +1, "#e0e0ff")
        self._jog_btn(g, "右 →", 1, 1, "j1", -1, "#e0e0ff")
        tk.Label(g, text="伸縮", font=("MS Gothic", 8), fg="gray").grid(row=0, column=2, columnspan=2)
        self._jog_btn(g, "伸ばす", 1, 2, "reach", +1, "#e0ffe0")
        self._jog_btn(g, "縮める", 1, 3, "reach", -1, "#e0ffe0")
        g2 = tk.Frame(f); g2.pack(padx=5, pady=3)
        tk.Label(g2, text="上下", font=("MS Gothic", 8), fg="gray").grid(row=0, column=0, columnspan=2)
        self._jog_btn_raw(g2, "↑ 上", 1, 0, 5)
        self._jog_btn_raw(g2, "↓ 下", 1, 1, 6)
        tk.Label(g2, text="R回転", font=("MS Gothic", 8), fg="blue").grid(row=0, column=2, columnspan=2)
        self._jog_btn_raw(g2, "R左回", 1, 2, 7, "#ffe0e0")
        self._jog_btn_raw(g2, "R右回", 1, 3, 8, "#ffe0e0")
        gf = tk.Frame(f); gf.pack(padx=5, pady=3)
        tk.Button(gf, text="開く", bg="green", fg="white", width=6, command=lambda: self.gripper_action(False)).pack(side="left", padx=2)
        tk.Button(gf, text="閉じる", bg="red", fg="white", width=6, command=lambda: self.gripper_action(True)).pack(side="left", padx=2)

        # ========== Step 0: キャリブレーション ==========
        f_cal = tk.LabelFrame(left, text="Step 0: ロボットアームキャリブ", font=("MS Gothic", 9, "bold"), fg="darkgreen")
        f_cal.pack(pady=2, fill="x")
        self.hand_calib_label = tk.Label(f_cal, 
            text=f"基準(R-J1) = {self.hand_base_diff:.1f}°" if self.hand_base_diff is not None else "基準 = 未設定",
            font=("Consolas", 9, "bold"),
            fg="green" if self.hand_base_diff is not None else "red")
        self.hand_calib_label.pack(padx=5, pady=1)
        tk.Button(f_cal, text="キャリブ保存 (現在=水平)",
                  command=self.save_hand_calibration,
                  bg="#228B22", fg="white",
                  font=("MS Gothic", 9), width=24).pack(pady=2)
        tk.Label(f_cal, text="ハンドを開き、カメラと水平にし、\n床につけた状態で押す",
                 font=("MS Gothic", 8), fg="gray", justify="center").pack()

        # ========== Step 1〜3 個別ボタン ==========
        f_steps = tk.LabelFrame(left, text="ステップ実行", font=("MS Gothic", 9, "bold"), fg="#333")
        f_steps.pack(pady=2, fill="x")

        step_frame = tk.Frame(f_steps)
        step_frame.pack(padx=3, pady=2, fill="x")

        self.search_btn = tk.Button(step_frame, text="Step1:\n探索",
                  command=self.toggle_search,
                  bg="#2266aa", fg="white",
                  font=("MS Gothic", 9), width=8, height=2)
        self.search_btn.pack(side="left", padx=2, pady=2)

        tk.Button(step_frame, text="Step2:\nR軸合わせ",
                  command=self.test_r_align,
                  bg="#7744aa", fg="white",
                  font=("MS Gothic", 9), width=10, height=2).pack(side="left", padx=2, pady=2)

        tk.Button(step_frame, text="Step3:\n把持",
                  command=self.test_grasp,
                  bg="#8B0000", fg="white",
                  font=("MS Gothic", 9), width=8, height=2).pack(side="left", padx=2, pady=2)

        self.status_label = tk.Label(f_steps, text="待機中", font=("Consolas", 10, "bold"), fg="gray")
        self.status_label.pack(pady=2)

        # ========== 自動探索 (Step1→2→3 一括) ==========
        f_auto = tk.LabelFrame(left, text="★ 自動探索 (Step1→2→3)", font=("MS Gothic", 10, "bold"), fg="darkblue")
        f_auto.pack(pady=3, fill="x")
        self.auto_btn = tk.Button(f_auto, text="▶ 自動探索",
                  command=self.run_full_auto,
                  bg="#114488", fg="white",
                  font=("MS Gothic", 11, "bold"), width=22, height=1)
        self.auto_btn.pack(pady=3)
        tk.Label(f_auto, text="探索→R軸合わせ→把持 を連続実行",
                 font=("MS Gothic", 8), fg="gray").pack()

        # 検出設定
        f = tk.LabelFrame(left, text="検出設定", font=("MS Gothic", 8))
        f.pack(pady=2, fill="x")
        tk.Label(f, text=f"H:{self.detector.h_min}-{self.detector.h_max} S:{self.detector.s_min}-{self.detector.s_max}", font=("Consolas", 8), fg="purple").pack(padx=5)

        # ★ TGT JSON対応: TGTキャリブ情報表示
        f_tgt = tk.LabelFrame(left, text="TGTキャリブ", font=("MS Gothic", 8))
        f_tgt.pack(pady=2, fill="x")
        tgt_text = (f"Z1={self.CALIB_Z1:.1f}→({self.CALIB_X1},{self.CALIB_Y1}) "
                    f"Z2={self.CALIB_Z2:.1f}→({self.CALIB_X2},{self.CALIB_Y2})")
        tk.Label(f_tgt, text=tgt_text, font=("Consolas", 7), fg="darkblue").pack(padx=3)

        # 右側
        right = tk.Frame(main)
        right.pack(side="left", padx=5, fill="both", expand=True)
        rf = tk.Frame(right); rf.pack(pady=3, fill="x")
        tk.Button(rf, text="カメラ開始", command=self.start_camera, bg="lightgreen", width=10).pack(side="left", padx=3)
        tk.Button(rf, text="カメラ停止", command=self.stop_camera, bg="gray", width=10).pack(side="left", padx=3)

        cf = tk.LabelFrame(right, text="カメラ映像")
        cf.pack(pady=3)
        self.camera_label = tk.Label(cf, width=640, height=480, bg="black")
        self.camera_label.pack()
        self.camera_label.bind("<Button-1>", self.on_camera_click)

        lf = tk.LabelFrame(right, text="ログ")
        lf.pack(pady=3, fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(lf, height=12, width=80, font=("Consolas", 8))
        self.log_text.pack(padx=3, pady=3, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(500, self._periodic_update)

    def on_camera_click(self, event):
        self.log(f"★ CLICK: (x={event.x}, y={event.y}) @ Z={self.current_z_for_calc:.1f}", "USER")

    def _jog_btn(self, parent, txt, r, c, mode, direction, color="#e0e0e0"):
        b = tk.Button(parent, text=txt, width=6, height=2, bg=color)
        b.grid(row=r, column=c, padx=2, pady=2)
        if mode == "j1": b.bind("<ButtonPress-1>", lambda e: self._start_jog_j1(direction))
        elif mode == "reach": b.bind("<ButtonPress-1>", lambda e: self._start_jog_reach(direction))
        b.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

    def _jog_btn_raw(self, parent, txt, r, c, cmd_id, color="#e0e0e0"):
        b = tk.Button(parent, text=txt, width=6, height=1, bg=color)
        b.grid(row=r, column=c, padx=2, pady=1)
        b.bind("<ButtonPress-1>", lambda e: self._start_jog_raw(cmd_id))
        b.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

    def _start_jog_j1(self, d): dType.SetJOGCmd(self.api, True, 1 if d > 0 else 2, isQueued=1)
    def _start_jog_reach(self, d): dType.SetJOGCmd(self.api, False, 1 if d > 0 else 2, isQueued=1)
    def _start_jog_raw(self, cid): dType.SetJOGCmd(self.api, False, cid, isQueued=1)
    def _stop_jog(self): dType.SetJOGCmd(self.api, False, 0, isQueued=1)

    def jog_move(self, cmd_id, duration, is_joint=False):
        if not self.is_connected: return
        dType.SetQueuedCmdClear(self.api)
        dType.SetJOGCmd(self.api, is_joint, cmd_id, isQueued=1)
        time.sleep(duration)
        dType.SetQueuedCmdClear(self.api)
        dType.SetJOGCmd(self.api, is_joint, 0, isQueued=1)
        time.sleep(0.1)

    def jog_j1(self, direction, duration):
        self.jog_move(1 if direction > 0 else 2, duration, is_joint=True)
    def jog_reach(self, direction, duration):
        self.jog_move(1 if direction > 0 else 2, duration, is_joint=False)
    def jog_z(self, direction, duration):
        self.jog_move(5 if direction > 0 else 6, duration, is_joint=False)

    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {msg}")
        self.root.after(0, lambda: self.log_text.insert(tk.END, f"[{ts}] [{level}] {msg}\n") or self.log_text.see(tk.END))

    def connect(self):
        if not DOBOT_AVAILABLE: return
        r = dType.ConnectDobot(self.api, "COM3", 115200)
        if r[0] == 0:
            self.is_connected = True
            dType.SetQueuedCmdClear(self.api)
            dType.SetQueuedCmdStartExec(self.api)
            dType.SetJOGCommonParams(self.api, 60, 60, isQueued=1)
            dType.SetJOGCoordinateParams(self.api, 60, 60, 60, 60, 60, 60, 60, 60, isQueued=1)
            dType.SetJOGJointParams(self.api, 60, 60, 60, 60, 60, 60, 60, 60, isQueued=1)
            dType.ClearAllAlarmsState(self.api)
            self.conn_label.config(text="接続済み ✓", fg="green")
            self.log("Dobot接続完了 (JOG速度=60)")
            self._update_pos()
        else: self.log(f"接続失敗 (code={r[0]})", "ERROR")

    def _clear_alarm(self):
        dType.ClearAllAlarmsState(self.api)
        self.conn_label.config(text="接続済み ✓", fg="green")

    def get_polar(self):
        if not self.is_connected: return None
        p = dType.GetPose(self.api)
        return (math.degrees(math.atan2(p[1], p[0])), math.sqrt(p[0]**2 + p[1]**2), p[2], p[3])

    def _update_pos(self):
        pol = self.get_polar()
        if pol:
            self.current_z_for_calc = pol[2] 
            self.pos_label.config(text=f"J1={pol[0]:7.1f}°  Reach={pol[1]:6.1f}mm\nZ ={pol[2]:7.1f}mm  R    ={pol[3]:7.1f}°")

    def _periodic_update(self):
        if self.is_connected: self._update_pos()
        self.root.after(1000, self._periodic_update)

    def gripper_action(self, close):
        if not self.is_connected: return
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        
        if close:
            dType.SetEndEffectorSuctionCupEx(self.api, True, True, isQueued=1)
            time.sleep(0.1)
            dType.SetEndEffectorGripperEx(self.api, True, True, isQueued=1)
            time.sleep(1.0)
            dType.SetEndEffectorSuctionCupEx(self.api, False, False, isQueued=1)
            dType.SetEndEffectorGripperEx(self.api, False, False, isQueued=1)
        else:
            dType.SetEndEffectorSuctionCupEx(self.api, True, True, isQueued=1)
            time.sleep(0.1)
            dType.SetEndEffectorGripperEx(self.api, True, False, isQueued=1)
            time.sleep(1.0)
            dType.SetEndEffectorSuctionCupEx(self.api, False, False, isQueued=1)
            dType.SetEndEffectorGripperEx(self.api, False, False, isQueued=1)
    
    def gripper_release(self):
        if not self.is_connected: return
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetEndEffectorSuctionCupEx(self.api, True, True, isQueued=1)
        time.sleep(0.1)
        dType.SetEndEffectorGripperEx(self.api, True, False, isQueued=1)
        time.sleep(1.0)
        dType.SetEndEffectorSuctionCupEx(self.api, False, False, isQueued=1)
        dType.SetEndEffectorGripperEx(self.api, False, False, isQueued=1)

    def start_camera(self):
        if self.is_camera_running: return
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_camera_running = True
        self.log("カメラ開始")
        self._update_camera()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap: self.cap.release(); self.cap = None

    def _get_dynamic_target(self):
        z = self.current_z_for_calc
        # ★ v10.2: TGTのZ値をfloor_zオフセット補正
        z1 = self._tgt_abs_z(self.CALIB_Z1)
        z2 = self._tgt_abs_z(self.CALIB_Z2)
        if abs(z2 - z1) < 0.001: return self.CALIB_X1, self.CALIB_Y1
        ratio = (z - z1) / (z2 - z1)
        tgt_x = self.CALIB_X1 + (self.CALIB_X2 - self.CALIB_X1) * ratio
        tgt_y = self.CALIB_Y1 + (self.CALIB_Y2 - self.CALIB_Y1) * ratio
        return int(tgt_x), int(tgt_y)

    def _update_camera(self):
        if not self.is_camera_running: return
        ret, frame = self.cap.read()
        if ret:
            det = self.detector.detect(frame)
            with self.frame_lock:
                self.current_frame = frame.copy()
                self.current_det = det
            display = frame.copy()
            self.detector.draw(display, det)
            
            tgt_x, tgt_y = self._get_dynamic_target()
            cv2.drawMarker(display, (tgt_x, tgt_y), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(display, f"TGT(Z={self.current_z_for_calc:.0f})", (tgt_x + 10, tgt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            if det:
                cx, cy = det['center']
                dx = cx - tgt_x
                dy = cy - tgt_y
                cv2.putText(display, f"dx:{dx:+d} dy:{dy:+d}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                self.last_detected_angle = det['short_angle']
                
                obj_ang = det['short_angle']
                long_ang_rad = np.radians(obj_ang + 90)
                L_obj = 80
                odx = int(L_obj * np.cos(long_ang_rad))
                ody = int(L_obj * np.sin(long_ang_rad))
                cv2.line(display, (cx - odx, cy - ody), (cx + odx, cy + ody), (0, 255, 0), 3)
                cv2.putText(display, f"OBJ:{obj_ang:+.1f}", (cx + odx + 5, cy + ody),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            h, w = display.shape[:2]
            hcx, hcy = w // 2, h // 2
            
            if self.hand_base_diff is not None and self.is_connected:
                pol = self.get_polar()
                if pol:
                    current_diff = pol[3] - pol[0]
                    hand_cam_angle = current_diff - self.hand_base_diff
                    
                    hand_rad = np.radians(hand_cam_angle)
                    hdx, hdy = int(120*np.cos(hand_rad)), int(120*-np.sin(hand_rad))
                    cv2.line(display, (320-hdx, 240-hdy), (320+hdx, 240+hdy), (0, 0, 255), 3)
                    cv2.putText(display, f"HAND:{hand_cam_angle:+.1f}", (hcx + hdx + 5, hcy + hdy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    
                    self.root.after(0, lambda a=hand_cam_angle: 
                        self.hand_calib_label.config(
                            text=f"基準(R-J1)={self.hand_base_diff:.1f}° | 現在:{a:+.1f}°",
                            fg="green"))
                    
                    if det:
                        obj_long = det['short_angle'] 
                        angle_diff = hand_cam_angle - obj_long
                        while angle_diff > 90: angle_diff -= 180
                        while angle_diff < -90: angle_diff += 180
                        
                        if abs(angle_diff) < 10:
                            color_judge = (0, 255, 0)
                            judge_text = "OK!"
                        elif abs(angle_diff) < 25:
                            color_judge = (0, 255, 255)
                            judge_text = "CLOSE"
                        else:
                            color_judge = (0, 0, 255)
                            judge_text = f"diff:{angle_diff:+.0f}"
                        
                        cv2.putText(display, f"CROSS: {judge_text}", (10, 440),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_judge, 2)

            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
        self.root.after(33, self._update_camera)

    def _get_detection(self):
        with self.frame_lock: return self.current_det

    # ======================== ★ v10.0: 統合キャリブレーション ========================
    def save_hand_calibration(self):
        if not self.is_connected:
            self.log("Dobot未接続", "WARN"); return
        
        p = dType.GetPose(self.api)
        r_now = p[3]
        j1_now = math.degrees(math.atan2(p[1], p[0]))
        z_now = p[2]
        
        self.hand_base_diff = r_now - j1_now
        self.floor_z = z_now
        
        data = {
            'hand_base_diff': round(self.hand_base_diff, 2),
            'floor_z': round(self.floor_z, 2),
            'r_at_save': round(r_now, 2),
            'j1_at_save': round(j1_now, 2),
            'z_at_save': round(z_now, 2),
            'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'note': 'ハンドを開き水平にして床につけた状態で保存。R-J1基準値と床Zを記録。',
        }
        
        fp = self._fp("hand_calibration.json")
        with open(fp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.log("=" * 50)
        self.log(f"★ キャリブレーション保存完了!")
        self.log(f"  基準(R-J1) = {self.hand_base_diff:.1f}°")
        self.log(f"  床Z = {self.floor_z:.1f}mm")
        self.log(f"  (R={r_now:.1f}°, J1={j1_now:.1f}°, Z={z_now:.1f}mm)")
        self.log(f"  実効Z値:")
        self.log(f"    Z_GRASP = {self._abs_z(self.Z_GRASP_REL):.1f}mm (床+{self.Z_GRASP_REL})")
        self.log(f"    Z_LIFT = {self._abs_z(self.Z_LIFT_REL):.1f}mm (床+{self.Z_LIFT_REL})")
        self.log("=" * 50)
        
        self.hand_calib_label.config(
            text=f"R-J1={self.hand_base_diff:.1f}° | 床Z={self.floor_z:.1f}mm", fg="green")

    # ======================== ★ v10.0: R軸角度合わせ (水平基準) ========================
    def _r_axis_alignment(self):
        self.log("=" * 60)
        self.log("★★★ R軸角度合わせ (v10.0 水平基準統合) ★★★")
        self.log("=" * 60)
        
        if self.hand_base_diff is None:
            self.log("[R-ALIGN] ⚠ ハンド基準が未設定！キャリブしてください", "ERROR")
            self._set_status("キャリブ未設定", "red")
            return False
        
        det = self._get_detection()
        if det is None:
            time.sleep(0.5)
            det = self._get_detection()
            if det is None:
                self.log("[R-ALIGN] ⚠ 物体が見えない！", "WARN")
                self._set_status("物体見えず", "red")
                return False
        
        obj_angle = det['short_angle']
        
        p = dType.GetPose(self.api)
        curr_r = p[3]
        j1_now = math.degrees(math.atan2(p[1], p[0]))
        
        self.log(f"[R-ALIGN] 物体角度: {obj_angle:.1f}°")
        self.log(f"[R-ALIGN] 基準(R-J1): {self.hand_base_diff:.1f}°")
        self.log(f"[R-ALIGN] 現在 J1={j1_now:.1f}° R={curr_r:.1f}°")
        
        target_r = j1_now + self.hand_base_diff + obj_angle 
        self.log(f"[R-ALIGN] target_R = {j1_now:.1f} + {self.hand_base_diff:.1f} + ({obj_angle:.1f}) + 90 = {target_r:.1f}°")
        
        adjust_count = 0
        while target_r > self.R_MAX and adjust_count < 4:
            target_r -= 180
            adjust_count += 1
        while target_r < self.R_MIN and adjust_count < 4:
            target_r += 180
            adjust_count += 1
        if adjust_count > 0:
            self.log(f"[R-ALIGN] ±180°補正 x{adjust_count} → target_R = {target_r:.1f}°")
        
        if target_r < self.R_MIN or target_r > self.R_MAX:
            self.log(f"[R-ALIGN] ⚠ 補正後も範囲外: {target_r:.1f}° → 中断", "ERROR")
            self._set_status("R範囲外", "red")
            return False
        
        self.log(f"[R-ALIGN] 現在R = {curr_r:.1f}° → 目標R = {target_r:.1f}° (差={target_r-curr_r:+.1f}°)")
        
        MODE_PTP_MOVL_XYZ = 1
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetPTPCmd(self.api, MODE_PTP_MOVL_XYZ, 
                       p[0], p[1], p[2], target_r, isQueued=1)
        time.sleep(3.0)
        
        p2 = dType.GetPose(self.api)
        self.log(f"[R-ALIGN] 完了: R={p2[3]:.1f}° (目標{target_r:.1f}°, 誤差{p2[3]-target_r:+.1f}°)")
        self.log(f"[R-ALIGN]   X={p2[0]:.1f} Y={p2[1]:.1f} Z={p2[2]:.1f}")
        
        time.sleep(0.5)
        det2 = self._get_detection()
        if det2:
            self.log(f"[R-ALIGN] 回転後の物体: angle={det2['short_angle']:.1f}° area={det2['area']:.0f}")
        else:
            self.log(f"[R-ALIGN] 回転後: 物体見えず（爪で隠れている可能性 → 正解に近い？）")
        
        self.log(f"★★★ R軸角度合わせ完了（目視確認してください） ★★★")
        self._set_status("★ 角度合わせ完了", "green")
        return True

    # ======================== R軸合わせ 単体テスト ========================
    def test_r_align(self):
        if not self.is_connected or not self.is_camera_running:
            self.log("ロボットまたはカメラ未接続", "WARN"); return
        if self.hand_base_diff is None:
            self.log("ハンド基準が未設定！キャリブしてください", "ERROR"); return
        
        def _run():
            self.log("=" * 60)
            self.log("★ R軸角度合わせテスト (v10.0 水平基準)")
            self.log(f"  基準(R-J1) = {self.hand_base_diff:.1f}°")
            self.log(f"  式: target_R = J1 + 基準 + obj_angle ")
            self.log("=" * 60)
            
            p = dType.GetPose(self.api)
            self.log(f"現在: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f} R={p[3]:.1f}")
            
            det = self._get_detection()
            if det:
                self.log(f"物体: angle={det['short_angle']:.1f}° center={det['center']} area={det['area']:.0f}")
            else:
                self.log("⚠ 物体未検出", "WARN")
            
            result = self._r_axis_alignment()
            self.log(f"テスト完了: {'成功' if result else '失敗'}")
        
        threading.Thread(target=_run, daemon=True).start()

    # ======================== 把持テスト ========================
    def test_grasp(self):
        if not self.is_connected or not self.is_camera_running:
            self.log("ロボットまたはカメラ未接続", "WARN"); return
        
        def _run():
            self.log("=" * 60)
            self.log("★★★ 把持テスト (降下→掴む→持上げ) ★★★")
            self.log("=" * 60)
            
            self.log("[GRASP] Step 1: Z降下（把持位置へ）")
            self._descend_to_grasp()
            
            self.log("[GRASP] Step 2: グリッパー閉じる")
            self.gripper_action(True)
            time.sleep(0.5)
            
            self.log("[GRASP] Step 3: 持ち上げ")
            self._lift_up()
            
            self.log("")
            self.log("=" * 60)
            self.log("★★★ 把持テスト完了 ★★★")
            self.log("  物体を掴めていますか？")
            self.log("  → 開くボタンで離す")
            self.log("=" * 60)
            self._set_status("★ 把持完了", "green")
        
        threading.Thread(target=_run, daemon=True).start()

    # ======================== 降下→把持→持上げ ========================
    Z_GRASP = -32.0
    Z_LIFT = 20.0
    
    def _descend_to_grasp(self):
        p = dType.GetPose(self.api)
        target_z = self._abs_z(self.Z_GRASP_REL)
        
        self.log(f"[DESCEND] Z={p[2]:.1f} → Z={target_z:.1f} (一気に降下)")
        
        MODE_PTP_MOVL_XYZ = 2
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetPTPCmd(self.api, MODE_PTP_MOVL_XYZ,
                       p[0], p[1], target_z, p[3], isQueued=1)
        time.sleep(4.0)
        
        p2 = dType.GetPose(self.api)
        self.log(f"[DESCEND] 降下完了: Z={p2[2]:.1f}")
    
    def _lift_up(self):
        p = dType.GetPose(self.api)
        target_z = self._abs_z(self.Z_LIFT_REL)
        self.log(f"[LIFT] 現在Z={p[2]:.1f} → 目標Z={target_z:.1f}")
        
        MODE_PTP_MOVL_XYZ = 2
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetPTPCmd(self.api, MODE_PTP_MOVL_XYZ,
                       p[0], p[1], target_z, p[3], isQueued=1)
        time.sleep(4.0)
        
        p2 = dType.GetPose(self.api)
        self.log(f"[LIFT] 持ち上げ完了: Z={p2[2]:.1f}")

    # ======================== 探索 ========================
    def toggle_search(self):
        if self.searching:
            self.stop_flag.set()
            self.searching = False
            self.search_btn.config(text="Step1: 探索", bg="#2266aa")
            self._set_status("停止", "orange")
        else:
            if not self.is_connected or not self.is_camera_running:
                self.log("ロボットまたはカメラ未接続", "WARN"); return
            if self.floor_z is None:
                self.log("床Z未設定！Step 0でキャリブしてください", "ERROR"); return
            self.searching = True
            self.stop_flag.clear()
            self.search_btn.config(text="■ 停止", bg="red")
            self._set_status("Step1: 探索中...", "blue")
            threading.Thread(target=self._search_worker, daemon=True).start()

    def run_full_auto(self):
        if self.searching:
            self.stop_flag.set()
            self.log("★ 緊急停止要求", "WARN")
            self._set_status("停止中...", "orange")
            return
        
        if not self.is_connected or not self.is_camera_running:
            self.log("ロボットまたはカメラ未接続", "WARN"); return
        if self.hand_base_diff is None:
            self.log("ハンド基準が未設定！Step 0でキャリブしてください", "ERROR"); return
        if self.floor_z is None:
            self.log("床Z未設定！Step 0でキャリブしてください", "ERROR"); return
        
        def _run():
            try:
                self.log("=" * 60)
                self.log("★★★ 自動探索開始 (Step1→2→3) ★★★")
                self.log("=" * 60)
                self.searching = True
                self.stop_flag.clear()
                self.root.after(0, lambda: self.auto_btn.config(text="■ 緊急停止", bg="#CC0000"))
                
                self._set_status("Step1: 探索中...", "blue")
                self.gripper_action(False)
                time.sleep(0.5)
                
                pol = self.get_polar()
                if pol: self.log(f"開始位置: J1={pol[0]:.1f}° Reach={pol[1]:.1f}mm")
                
                found = self._jog_scan()
                if not found or self.stop_flag.is_set():
                    if not self.stop_flag.is_set():
                        self.log("物体が見つかりませんでした")
                        self._set_status("未検出", "red")
                    return
                
                self._set_status("Step1: キャリブ計測...", "orange")
                self._measure_calib()
                if self.stop_flag.is_set(): return
                
                self._set_status("Step1: 粗動サーボ...", "blue")
                if not self._jog_servo_proportional():
                    self._set_status("粗動失敗", "red"); return
                if self.stop_flag.is_set(): return
                
                self._set_status("Step1: 微調整...", "purple")
                if not self._jog_servo_fine():
                    self._set_status("微調整失敗", "red"); return
                if self.stop_flag.is_set(): return
                
                self._set_status("Step1: Z降下...", "blue")
                self._descend_with_servo()
                if self.stop_flag.is_set(): return
                
                self._set_status("Step2: R軸合わせ...", "#7744aa")
                r_ok = self._r_axis_alignment()
                if not r_ok:
                    self._set_status("R軸合わせ失敗", "red"); return
                if self.stop_flag.is_set(): return
                
                self._set_status("Step3: 降下中...", "#8B0000")
                self.log("[AUTO] Z降下（把持位置へ）")
                self._descend_to_grasp()
                if self.stop_flag.is_set(): return
                
                self._set_status("Step3: 把持中...", "#8B0000")
                self.log("[AUTO] グリッパー閉じる")
                self.gripper_action(True)
                time.sleep(0.5)
                if self.stop_flag.is_set(): return
                
                self.log("[AUTO] 持ち上げ")
                self._lift_up()
                
                self.log("=" * 60)
                self.log("★★★ 自動探索完了！ ★★★")
                self.log("=" * 60)
                self._set_status("★ 自動探索完了", "green")
                
            except Exception as e:
                self.log(f"自動探索エラー: {e}", "ERROR")
                import traceback; traceback.print_exc()
            finally:
                self.searching = False
                if self.stop_flag.is_set():
                    self._set_status("★ 停止済み", "orange")
                    self.log("★ 緊急停止完了")
                self.root.after(0, lambda: self.auto_btn.config(text="▶ 自動探索", bg="#114488"))
        
        threading.Thread(target=_run, daemon=True).start()

    def _set_status(self, text, color):
        self.root.after(0, lambda: self.status_label.config(text=text, fg=color))

    # ======================== ★ v10.0: サーボ降下 (ステップ最適化) ========================
    Z_DESCENT_STEPS = [30.0, 10.0, 0.0]
    DESCENT_SERVO_PX = 20
    DESCENT_SETTLE = 0.6
    DESCENT_SERVO_MAX = 8

    def _descend_with_servo(self):
        self.log("=" * 60)
        descent_abs = [self._abs_z(z) for z in self.Z_DESCENT_REL]
        self.log(f"★★★ サーボ付きZ降下 (床相対={self.Z_DESCENT_REL}, 絶対={descent_abs}) ★★★")
        
        p = dType.GetPose(self.api)
        self.log(f"[DESCEND_SERVO] 開始位置: Z={p[2]:.1f}mm")
        
        for z_rel in self.Z_DESCENT_REL:
            z_target = self._abs_z(z_rel)
            if self.stop_flag.is_set():
                return
            
            p = dType.GetPose(self.api)
            if p[2] <= z_target:
                self.log(f"[DESCEND_SERVO] Z={z_target}mm はスキップ (現在Z={p[2]:.1f})")
                continue
            
            self.log(f"[DESCEND_SERVO] ▼ Z={p[2]:.1f} → Z={z_target}mm")
            self.current_z_for_calc = z_target
            dType.SetQueuedCmdClear(self.api)
            dType.SetQueuedCmdStartExec(self.api)
            dType.SetPTPCmd(self.api, 2, p[0], p[1], z_target, p[3], isQueued=1)
            time.sleep(3.0)
            
            time.sleep(self.DESCENT_SETTLE)
            self._update_pos()
            
            self.log(f"[DESCEND_SERVO] ▷ Z={z_target}mm でサーボ補正...")
            for i in range(self.DESCENT_SERVO_MAX):
                if self.stop_flag.is_set():
                    return
                time.sleep(self.SETTLE_TIME)
                det = self._get_detection()
                if det is None:
                    self.log(f"[DESCEND_SERVO]   iter{i}: 物体未検出 → スキップ")
                    continue
                
                cx, cy = det['center']
                tgt_x, tgt_y = self._get_dynamic_target()
                dx = cx - tgt_x
                dy = cy - tgt_y
                self.log(f"[DESCEND_SERVO]   iter{i}: dx={dx:+.0f} dy={dy:+.0f}")
                
                if abs(dx) < self.DESCENT_SERVO_PX and abs(dy) < self.DESCENT_SERVO_PX:
                    self.log(f"[DESCEND_SERVO]   ★ 収束! (Z={z_target}mm)")
                    break
                
                if abs(dy) >= self.DESCENT_SERVO_PX:
                    j1_dir = -1 if dy > 0 else +1
                    t = self.JOG_SERVO_LONG if abs(dy) > 60 else self.JOG_SERVO_TIME
                    self.jog_j1(j1_dir, t)
                    time.sleep(0.1)
                
                if abs(dx) >= self.DESCENT_SERVO_PX:
                    reach_dir = +1 if dx > 0 else -1
                    t = self.JOG_SERVO_LONG if abs(dx) > 60 else self.JOG_SERVO_TIME
                    self.jog_reach(reach_dir, t)
                    time.sleep(0.1)
        
        p2 = dType.GetPose(self.api)
        self.log(f"[DESCEND_SERVO] 完了: X={p2[0]:.1f} Y={p2[1]:.1f} Z={p2[2]:.1f}")
        self.log("★★★ サーボ付き降下完了 ★★★")

    def _search_worker(self):
        try:
            self.log("=" * 60)
            self.log("===== JOG探索開始 (v10.1) =====")
            self.gripper_action(False)
            time.sleep(0.5)
            
            pol = self.get_polar()
            if pol: self.log(f"開始位置: J1={pol[0]:.1f}° Reach={pol[1]:.1f}mm")

            found = self._jog_scan()
            if not found:
                if not self.stop_flag.is_set():
                    self.log("物体が見つかりませんでした")
                    self._set_status("未検出", "red")
                self._finish_search(); return

            self._set_status("キャリブ計測中...", "orange")
            self._measure_calib()

            self._set_status("粗動サーボ中...", "blue")
            success = self._jog_servo_proportional()
            
            if success:
                self.log("★★★ 粗動サーボ完了！ ★★★")
                self._set_status("微調整中...", "purple")
                success2 = self._jog_servo_fine()
                if success2:
                    self.log("★★★ 微調整完了！ ★★★")
                    self._descend_with_servo()
                    self._set_status("★ Z=0到達 完了", "green")
                else:
                    self.log("微調整サーボ収束せず")
                    self._set_status("微調整失敗", "red")
            else:
                self.log("粗動サーボ収束せず")
                self._set_status("粗動失敗", "red")
        except Exception as e:
            self.log(f"探索エラー: {e}", "ERROR")
            import traceback; traceback.print_exc()
        finally: self._finish_search()

    def _finish_search(self):
        self.searching = False
        self.root.after(0, lambda: self.search_btn.config(text="▶ 探索開始", bg="#2266aa"))

    def _jog_scan(self):
        self.log("[SCAN] J1をJOGで右回転しながら探索...")
        MAX_STEPS = 80; consecutive = 0; CONFIRM = 2
        for step in range(MAX_STEPS):
            if self.stop_flag.is_set(): return False
            pol = self.get_polar()
            j1_now = pol[0] if pol else 0
            if self.scan_j1_min is not None and j1_now < self.scan_j1_min:
                self.log(f"[SCAN] 探索範囲外 J1={j1_now:.1f}°"); return False
            self.jog_j1(-1, 0.20)
            time.sleep(self.SETTLE_TIME)
            det = self._get_detection()
            if det:
                consecutive += 1
                self.log(f"[SCAN] step {step}: 検出! ({consecutive}/{CONFIRM})")
                if consecutive >= CONFIRM: return True
            else: consecutive = 0
        return False

    def _measure_calib(self):
        self.log("[CALIB] ===== キャリブレーション計測開始 =====")

        time.sleep(self.SETTLE_TIME)
        det_before = self._get_detection()
        if det_before:
            cy_before = det_before['center'][1]
            self.log(f"[CALIB] J1: 計測前 cy={cy_before}")
            self.jog_j1(-1, self.CALIB_MOVE_TIME)
            time.sleep(self.SETTLE_TIME)
            det_after = self._get_detection()
            if det_after:
                cy_after = det_after['center'][1]
                delta_px = abs(cy_after - cy_before)
                self.log(f"[CALIB] J1: Δcy={delta_px:.1f}px / {self.CALIB_MOVE_TIME}秒")
                if delta_px >= self.CALIB_MIN_PX:
                    self.calib_j1_sec_per_px = self.CALIB_MOVE_TIME / delta_px
                    self.log(f"[CALIB] J1: ★ {self.calib_j1_sec_per_px*1000:.2f} ms/px")
            self.jog_j1(+1, self.CALIB_MOVE_TIME)
            time.sleep(self.SETTLE_TIME)

        time.sleep(self.SETTLE_TIME)
        det_before = self._get_detection()
        if det_before:
            cx_before = det_before['center'][0]
            self.log(f"[CALIB] Reach: 計測前 cx={cx_before}")
            self.jog_reach(-1, self.CALIB_MOVE_TIME)
            time.sleep(self.SETTLE_TIME)
            det_after = self._get_detection()
            if det_after:
                cx_after = det_after['center'][0]
                delta_px = abs(cx_after - cx_before)
                self.log(f"[CALIB] Reach: Δcx={delta_px:.1f}px / {self.CALIB_MOVE_TIME}秒")
                if delta_px >= self.CALIB_MIN_PX:
                    self.calib_reach_sec_per_px = self.CALIB_MOVE_TIME / delta_px
                    self.log(f"[CALIB] Reach: ★ {self.calib_reach_sec_per_px*1000:.2f} ms/px")
            self.jog_reach(+1, self.CALIB_MOVE_TIME)
            time.sleep(self.SETTLE_TIME)

        self.log("[CALIB] ===== キャリブレーション計測完了 =====")

    def _calc_move_time(self, px_error, axis, move_count):
        gain = self.GAIN_SCHEDULE.get(move_count, self.GAIN_DEFAULT)
        if axis == "j1" and self.calib_j1_sec_per_px is not None:
            raw_time = abs(px_error) * self.calib_j1_sec_per_px
            move_time = max(self.PROP_MIN_TIME, min(self.PROP_MAX_TIME, raw_time * gain))
            self.log(f"[PROP] J1: {abs(px_error):.0f}px × {self.calib_j1_sec_per_px*1000:.2f}ms/px × gain{gain:.2f} = {move_time*1000:.0f}ms")
        elif axis == "reach" and self.calib_reach_sec_per_px is not None:
            raw_time = abs(px_error) * self.calib_reach_sec_per_px
            move_time = max(self.PROP_MIN_TIME, min(self.PROP_MAX_TIME, raw_time * gain))
            self.log(f"[PROP] Reach: {abs(px_error):.0f}px × {self.calib_reach_sec_per_px*1000:.2f}ms/px × gain{gain:.2f} = {move_time*1000:.0f}ms")
        else:
            move_time = self.JOG_SERVO_LONG if abs(px_error) > 100 else self.JOG_SERVO_TIME
            self.log(f"[PROP] {axis}: キャリブなし → フォールバック {move_time*1000:.0f}ms")
        return move_time

    def _jog_servo_proportional(self):
        self.log(f"[PROP_SERVO] 比例制御粗動開始 (閾値={self.COARSE_PX}px)")
        MAX_ITER = 30; lost_count = 0; MAX_LOST = 8
        j1_move_count = 0; reach_move_count = 0

        for i in range(MAX_ITER):
            if self.stop_flag.is_set(): return False
            time.sleep(self.SETTLE_TIME)
            det = self._get_detection()
            if det is None:
                lost_count += 1
                if lost_count >= MAX_LOST: return False
                continue
            lost_count = 0
            cx, cy = det['center']
            tgt_x, tgt_y = self._get_dynamic_target()
            dx = cx - tgt_x; dy = cy - tgt_y
            self.log(f"[PROP_SERVO] iter {i}: dx={dx:+.0f} dy={dy:+.0f}")
            if abs(dx) < self.COARSE_PX and abs(dy) < self.COARSE_PX:
                self.log(f"[PROP_SERVO] ★ 粗動収束!"); return True

            if abs(dy) >= self.COARSE_PX:
                j1_move_count += 1
                j1_dir = -1 if dy > 0 else +1
                t = self._calc_move_time(dy, "j1", j1_move_count)
                self.log(f"[PROP_SERVO] J1: dy={dy:+.0f} → dir={j1_dir} t={t*1000:.0f}ms")
                self.jog_j1(j1_dir, t)
                time.sleep(0.1)

            if abs(dx) >= self.COARSE_PX:
                reach_move_count += 1
                reach_dir = +1 if dx > 0 else -1
                t = self._calc_move_time(dx, "reach", reach_move_count)
                self.log(f"[PROP_SERVO] Reach: dx={dx:+.0f} → dir={reach_dir} t={t*1000:.0f}ms")
                self.jog_reach(reach_dir, t)
                time.sleep(0.1)

        return False

    def _jog_servo_fine(self):
        self.log(f"[FINE_SERVO] 微調整開始 (閾値={self.CONVERGE_PX}px)")
        lost_count = 0; MAX_LOST = 10; self.last_move = None

        for i in range(self.SERVO_MAX_ITER):
            if self.stop_flag.is_set(): return False
            time.sleep(self.SETTLE_TIME)
            det = self._get_detection()
            if det is None:
                lost_count += 1
                if lost_count >= 2 and self.last_move:
                    mode, direction = self.last_move
                    if mode == "j1": self.jog_j1(-direction, 0.05)
                    else: self.jog_reach(-direction, 0.05)
                    time.sleep(0.2)
                if lost_count >= MAX_LOST: return False
                continue
            lost_count = 0
            cx, cy = det['center']
            tgt_x, tgt_y = self._get_dynamic_target()
            dx = cx - tgt_x; dy = cy - tgt_y
            self.log(f"[FINE_SERVO] iter {i}: dx={dx:+.0f} dy={dy:+.0f}")
            if abs(dx) < self.CONVERGE_PX and abs(dy) < self.CONVERGE_PX:
                self.log(f"[FINE_SERVO] ★ 微調整収束!"); return True

            if abs(dy) >= self.CONVERGE_PX:
                j1_dir = -1 if dy > 0 else +1
                t = self.JOG_SERVO_LONG if abs(dy) > 100 else self.JOG_SERVO_TIME
                self.log(f"[FINE_SERVO] J1: dy={dy:+.0f} → dir={j1_dir} t={t*1000:.0f}ms")
                self.jog_j1(j1_dir, t)
                self.last_move = ("j1", j1_dir)
                time.sleep(0.1)

            if abs(dx) >= self.CONVERGE_PX:
                reach_dir = +1 if dx > 0 else -1
                t = self.JOG_SERVO_LONG if abs(dx) > 100 else self.JOG_SERVO_TIME
                self.log(f"[FINE_SERVO] Reach: dx={dx:+.0f} → dir={reach_dir} t={t*1000:.0f}ms")
                self.jog_reach(reach_dir, t)
                self.last_move = ("reach", reach_dir)
                time.sleep(0.1)

        return False

    def _descend_to_approach(self):
        self.log(f"★★★ Phase 3: Z降下 → Z={self.Z_APPROACH}mm ★★★")
        p = dType.GetPose(self.api)
        self.log(f"[DESCEND] 現在: X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f} R={p[3]:.1f}")
        dType.SetQueuedCmdClear(self.api)
        dType.SetQueuedCmdStartExec(self.api)
        dType.SetPTPCmd(self.api, 2, p[0], p[1], self.Z_APPROACH, p[3], isQueued=1)
        time.sleep(5.0)
        p2 = dType.GetPose(self.api)
        self.log(f"[DESCEND] 降下完了: X={p2[0]:.1f} Y={p2[1]:.1f} Z={p2[2]:.1f} R={p2[3]:.1f}")

    def on_closing(self):
        self.stop_flag.set()
        self.stop_camera()
        if self.is_connected and DOBOT_AVAILABLE:
            try: dType.DisconnectDobot(self.api)
            except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x750")
    app = ObjectSearchApp(root)
    root.mainloop()