#!/usr/bin/env python3
"""
==============================================================================
TGT キャリブレーションツール v2
==============================================================================
Eye-in-Hand構成のDobot Magician用。
ステップ式UIで、TGTの線形補間パラメータを tgt_calibration.json に保存する。

手順:
  Step 1: グリッパーの先端を目印(×印)に合わせ、床につける
          → 「床の位置を記録」ボタンで floor_z を記録
  Step 2: カメラ画面上の目印(×印)をクリック → 点1(低い方)を記録
  Step 3: 「床+110mmへ移動」ボタンでアームを上げる (XYはそのまま)
          → カメラ画面上の同じ目印をクリック → 点2(高い方)を記録
  Step 4: 「JSONに保存」で tgt_calibration.json を出力
==============================================================================
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
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


class TgtCalibrationTool:
    """TGTキャリブレーション用GUIツール v2"""

    Z_UP_FROM_FLOOR = 110.0  # 点2の高さ: 床+110mm

    def __init__(self, root):
        self.root = root
        self.root.title("TGT キャリブレーションツール v2")

        # Dobot
        self.api = dType.load() if DOBOT_AVAILABLE else None
        self.is_connected = False

        # カメラ
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # キャリブデータ
        self.floor_z = None       # 床のZ座標（Dobot絶対値）
        self.recording_point = None  # "point1" or "point2" or None
        self.point1 = None        # (x, y, z)  z=floor_z
        self.point2 = None        # (x, y, z)  z=floor_z+110
        self.click_history = []

        # ステップ管理
        self.current_step = 0     # 0=未開始, 1=床記録待ち, 2=点1クリック待ち, 3=点2クリック待ち, 4=保存待ち

        self.setup_ui()
        self._update_step_display()

    def _fp(self, name):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def setup_ui(self):
        main = tk.Frame(self.root)
        main.pack(padx=5, pady=5, fill="both", expand=True)

        # ===== 左パネル =====
        left = tk.Frame(main)
        left.pack(side="left", padx=5, fill="y")

        # --- 接続 ---
        f = tk.LabelFrame(left, text="接続", font=("MS Gothic", 9, "bold"))
        f.pack(pady=3, fill="x")
        r = tk.Frame(f); r.pack(fill="x", padx=3, pady=3)
        tk.Button(r, text="Connect", command=self.connect,
                  bg="lightblue", width=10).pack(side="left", padx=2)
        tk.Button(r, text="ALM Reset", command=self._clear_alarm,
                  bg="#ff6666", fg="white", width=9).pack(side="left", padx=2)
        self.conn_label = tk.Label(f, text="未接続", fg="gray")
        self.conn_label.pack()

        # --- 現在位置 ---
        f = tk.LabelFrame(left, text="現在位置", font=("MS Gothic", 9))
        f.pack(pady=3, fill="x")
        self.pos_label = tk.Label(f, text="---", font=("Consolas", 9),
                                  justify="left", anchor="w")
        self.pos_label.pack(padx=5, pady=3, fill="x")

        # --- 手動操作 ---
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
        tk.Button(gf, text="開く", bg="green", fg="white", width=6,
                  command=lambda: self.gripper_action(False)).pack(side="left", padx=2)
        tk.Button(gf, text="閉じる", bg="red", fg="white", width=6,
                  command=lambda: self.gripper_action(True)).pack(side="left", padx=2)

        # ===== ★ ステップ式キャリブレーション =====
        f_cal = tk.LabelFrame(left, text="★ TGTキャリブレーション",
                               font=("MS Gothic", 10, "bold"), fg="darkblue")
        f_cal.pack(pady=5, fill="x")

        # --- Step 1: 床記録 ---
        self.f_step1 = tk.LabelFrame(f_cal, text="Step 1: 床の位置を記録",
                                      font=("MS Gothic", 9, "bold"), fg="#333")
        self.f_step1.pack(padx=5, pady=3, fill="x")
        tk.Label(self.f_step1,
                 text="グリッパーの先端を目印(×)に合わせ、\n床につけてください",
                 font=("MS Gothic", 8), fg="gray", justify="left").pack(padx=5, anchor="w")
        self.btn_floor = tk.Button(self.f_step1,
                  text="★ 床の位置を記録",
                  command=self.record_floor,
                  bg="#cc6600", fg="white",
                  font=("MS Gothic", 9, "bold"), width=22)
        self.btn_floor.pack(pady=3)
        self.lbl_floor = tk.Label(self.f_step1, text="未記録",
                                   font=("Consolas", 9), fg="gray")
        self.lbl_floor.pack()

        # --- Step 2: 点1クリック ---
        self.f_step2 = tk.LabelFrame(f_cal, text="Step 2: 床位置の目印をクリック",
                                      font=("MS Gothic", 9, "bold"), fg="#333")
        self.f_step2.pack(padx=5, pady=3, fill="x")
        tk.Label(self.f_step2,
                 text="カメラ画面上の目印(×)をクリックしてください\n（床にいる状態のまま）",
                 font=("MS Gothic", 8), fg="gray", justify="left").pack(padx=5, anchor="w")
        self.lbl_p1 = tk.Label(self.f_step2, text="未記録",
                                font=("Consolas", 9), fg="gray")
        self.lbl_p1.pack()

        # --- Step 3: 移動 + 点2クリック ---
        self.f_step3 = tk.LabelFrame(f_cal,
                  text=f"Step 3: 床+{self.Z_UP_FROM_FLOOR:.0f}mmで目印をクリック",
                  font=("MS Gothic", 9, "bold"), fg="#333")
        self.f_step3.pack(padx=5, pady=3, fill="x")
        tk.Label(self.f_step3,
                 text=f"まず「上へ移動」でアームを上げてから、\nカメラ画面上の同じ目印(×)をクリック",
                 font=("MS Gothic", 8), fg="gray", justify="left").pack(padx=5, anchor="w")
        self.btn_move_up = tk.Button(self.f_step3,
                  text=f"▲ 床+{self.Z_UP_FROM_FLOOR:.0f}mmへ移動",
                  command=self.move_to_z2,
                  bg="#4477aa", fg="white",
                  font=("MS Gothic", 9), width=22)
        self.btn_move_up.pack(pady=2)
        self.lbl_p2 = tk.Label(self.f_step3, text="未記録",
                                font=("Consolas", 9), fg="gray")
        self.lbl_p2.pack()

        # --- Step 4: 保存 ---
        self.f_step4 = tk.LabelFrame(f_cal, text="Step 4: 保存",
                                      font=("MS Gothic", 9, "bold"), fg="#333")
        self.f_step4.pack(padx=5, pady=3, fill="x")
        fb = tk.Frame(self.f_step4); fb.pack(padx=5, pady=3, fill="x")
        tk.Button(fb, text="リセット", command=self.reset_all,
                  bg="gray", fg="white", width=8).pack(side="left", padx=2)
        self.btn_save = tk.Button(fb, text="★ JSONに保存",
                  command=self.save_json,
                  bg="#006400", fg="white",
                  font=("MS Gothic", 10, "bold"), width=16)
        self.btn_save.pack(side="left", padx=5)

        # 全体ステータス
        self.overall_status = tk.Label(f_cal, text="",
                                        font=("MS Gothic", 9, "bold"), fg="gray")
        self.overall_status.pack(pady=3)

        # ===== 右パネル =====
        right = tk.Frame(main)
        right.pack(side="left", padx=5, fill="both", expand=True)

        rf = tk.Frame(right); rf.pack(pady=3, fill="x")
        tk.Button(rf, text="カメラ開始", command=self.start_camera,
                  bg="lightgreen", width=10).pack(side="left", padx=3)
        tk.Button(rf, text="カメラ停止", command=self.stop_camera,
                  bg="gray", width=10).pack(side="left", padx=3)

        cf = tk.LabelFrame(right, text="カメラ映像（クリックで座標記録）")
        cf.pack(pady=3)
        self.camera_label = tk.Label(cf, width=640, height=480, bg="black")
        self.camera_label.pack()
        self.camera_label.bind("<Button-1>", self.on_camera_click)

        lf = tk.LabelFrame(right, text="ログ")
        lf.pack(pady=3, fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(lf, height=10, width=80,
                                                   font=("Consolas", 8))
        self.log_text.pack(padx=3, pady=3, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(500, self._periodic_update)

    # ======================== ステップ表示管理 ========================
    def _update_step_display(self):
        """現在のステップに応じてUIの有効/無効・色を切り替え"""
        # 全ステップのフレームの色をリセット
        for f, step_num in [(self.f_step1, 1), (self.f_step2, 2),
                            (self.f_step3, 3), (self.f_step4, 4)]:
            if step_num < self.current_step:
                f.config(fg="green")  # 完了
            elif step_num == self.current_step:
                f.config(fg="darkblue")  # 現在
            else:
                f.config(fg="#999")  # 未到達

        # ボタン有効/無効
        floor_ok = self.floor_z is not None
        p1_ok = self.point1 is not None
        p2_ok = self.point2 is not None

        self.btn_floor.config(state="normal" if self.is_connected else "disabled")
        self.btn_move_up.config(state="normal" if (floor_ok and p1_ok and self.is_connected) else "disabled")
        self.btn_save.config(state="normal" if (p1_ok and p2_ok) else "disabled")

        # 全体ステータス
        if p1_ok and p2_ok:
            self.overall_status.config(text="✓ 準備完了 → 「JSONに保存」で完了！", fg="darkgreen")
        elif p1_ok and floor_ok:
            self.overall_status.config(text="→ Step 3: 上へ移動して目印をクリック", fg="blue")
        elif floor_ok:
            self.overall_status.config(text="→ Step 2: カメラ画面の目印をクリック", fg="blue")
        else:
            self.overall_status.config(text="→ Step 1: 床に降ろして記録してください", fg="orange")

    # ======================== Step 1: 床記録 ========================
    def record_floor(self):
        if not self.is_connected:
            self.log("Dobot未接続", "WARN"); return

        p = dType.GetPose(self.api)
        self.floor_z = p[2]

        self.log("=" * 50)
        self.log(f"★ 床の位置を記録: Z={self.floor_z:.1f}mm")
        self.log(f"  (X={p[0]:.1f} Y={p[1]:.1f} R={p[3]:.1f})")
        self.log(f"  → 点2の高さ: Z={self.floor_z + self.Z_UP_FROM_FLOOR:.1f}mm (床+{self.Z_UP_FROM_FLOOR:.0f})")
        self.log("=" * 50)

        self.lbl_floor.config(
            text=f"床Z = {self.floor_z:.1f}mm ✓",
            fg="green", font=("Consolas", 9, "bold"))
        self.btn_floor.config(bg="#228B22")

        self.current_step = 2
        self.recording_point = "point1"
        self.log("→ Step 2: カメラ画面上の目印(×)をクリックしてください")
        self._update_step_display()

    # ======================== Step 3: Z2へ移動 ========================
    def move_to_z2(self):
        if not self.is_connected or self.floor_z is None:
            self.log("Dobot未接続 or 床Z未記録", "WARN"); return

        target_z = self.floor_z + self.Z_UP_FROM_FLOOR

        def _run():
            p = dType.GetPose(self.api)
            self.log(f"Z移動: {p[2]:.1f} → {target_z:.1f}mm (床+{self.Z_UP_FROM_FLOOR:.0f})")
            dType.SetQueuedCmdClear(self.api)
            dType.SetQueuedCmdStartExec(self.api)
            dType.SetPTPCmd(self.api, 2, p[0], p[1], target_z, p[3], isQueued=1)
            time.sleep(4.0)
            p2 = dType.GetPose(self.api)
            self.log(f"移動完了: Z={p2[2]:.1f}mm")
            self.log("→ カメラ画面上の同じ目印(×)をクリックしてください")

        threading.Thread(target=_run, daemon=True).start()

    # ======================== カメラクリック ========================
    def on_camera_click(self, event):
        x, y = event.x, event.y
        self.click_history.append((x, y))
        self.log(f"CLICK: ({x}, {y})")

        if self.recording_point == "point1":
            if self.floor_z is None:
                self.log("⚠ 先にStep 1で床を記録してください", "WARN")
                return
            z = self.floor_z
            self.point1 = (x, y, z)
            self.lbl_p1.config(
                text=f"点1: ({x}, {y}) @ Z={z:.1f} (床) ✓",
                fg="green", font=("Consolas", 9, "bold"))
            self.recording_point = "point2"
            self.current_step = 3
            self.log(f"★ 点1 記録: px=({x}, {y}), Z={z:.1f}mm (床)")
            self.log(f"→ Step 3: 「▲ 床+{self.Z_UP_FROM_FLOOR:.0f}mmへ移動」を押してから目印をクリック")
            self._update_step_display()

        elif self.recording_point == "point2":
            if self.floor_z is None:
                return
            z = self.floor_z + self.Z_UP_FROM_FLOOR
            self.point2 = (x, y, z)
            self.lbl_p2.config(
                text=f"点2: ({x}, {y}) @ Z={z:.1f} (床+{self.Z_UP_FROM_FLOOR:.0f}) ✓",
                fg="green", font=("Consolas", 9, "bold"))
            self.recording_point = None
            self.current_step = 4
            self.log(f"★ 点2 記録: px=({x}, {y}), Z={z:.1f}mm (床+{self.Z_UP_FROM_FLOOR:.0f})")

            self.log("=" * 50)
            self.log(f"  点1: ({self.point1[0]}, {self.point1[1]}) @ Z={self.point1[2]:.1f} (床)")
            self.log(f"  点2: ({self.point2[0]}, {self.point2[1]}) @ Z={self.point2[2]:.1f} (床+{self.Z_UP_FROM_FLOOR:.0f})")
            self.log(f"  → 「JSONに保存」を押してください")
            self.log("=" * 50)
            self._update_step_display()

    # ======================== リセット ========================
    def reset_all(self):
        self.floor_z = None
        self.point1 = None
        self.point2 = None
        self.recording_point = None
        self.click_history.clear()
        self.current_step = 0

        self.lbl_floor.config(text="未記録", fg="gray", font=("Consolas", 9))
        self.lbl_p1.config(text="未記録", fg="gray", font=("Consolas", 9))
        self.lbl_p2.config(text="未記録", fg="gray", font=("Consolas", 9))
        self.btn_floor.config(bg="#cc6600")

        self._update_step_display()
        self.log("全データリセット")

    # ======================== JSON保存 ========================
    def save_json(self):
        if self.point1 is None or self.point2 is None:
            messagebox.showwarning("未完了", "点1と点2の両方を記録してください")
            return

        data = {
            "calib_z1": self.point1[2],
            "calib_x1": self.point1[0],
            "calib_y1": self.point1[1],
            "calib_z2": self.point2[2],
            "calib_x2": self.point2[0],
            "calib_y2": self.point2[1],
            "floor_z": self.floor_z,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": f"TGT calibration: Z1=floor({self.floor_z:.1f}), Z2=floor+{self.Z_UP_FROM_FLOOR:.0f}mm"
        }

        fp = self._fp("tgt_calibration.json")
        with open(fp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.log("=" * 60)
        self.log("★★★ TGTキャリブレーション保存完了! ★★★")
        self.log(f"  ファイル: {fp}")
        self.log(f"  床Z: {self.floor_z:.1f}mm")
        self.log(f"  点1: Z={data['calib_z1']:.1f} (床) → px=({data['calib_x1']}, {data['calib_y1']})")
        self.log(f"  点2: Z={data['calib_z2']:.1f} (床+{self.Z_UP_FROM_FLOOR:.0f}) → px=({data['calib_x2']}, {data['calib_y2']})")
        self.log("=" * 60)

        messagebox.showinfo("保存完了",
            f"tgt_calibration.json を保存しました\n\n"
            f"床Z: {self.floor_z:.1f}mm\n"
            f"点1: Z={data['calib_z1']:.1f} → ({data['calib_x1']}, {data['calib_y1']})\n"
            f"点2: Z={data['calib_z2']:.1f} → ({data['calib_x2']}, {data['calib_y2']})")

    # ======================== JOG・接続 ========================
    def _jog_btn(self, parent, txt, r, c, mode, direction, color="#e0e0e0"):
        b = tk.Button(parent, text=txt, width=6, height=2, bg=color)
        b.grid(row=r, column=c, padx=2, pady=2)
        if mode == "j1":
            b.bind("<ButtonPress-1>", lambda e: self._start_jog_j1(direction))
        elif mode == "reach":
            b.bind("<ButtonPress-1>", lambda e: self._start_jog_reach(direction))
        b.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

    def _jog_btn_raw(self, parent, txt, r, c, cmd_id, color="#e0e0e0"):
        b = tk.Button(parent, text=txt, width=6, height=1, bg=color)
        b.grid(row=r, column=c, padx=2, pady=1)
        b.bind("<ButtonPress-1>", lambda e: self._start_jog_raw(cmd_id))
        b.bind("<ButtonRelease-1>", lambda e: self._stop_jog())

    def _start_jog_j1(self, d):
        if self.is_connected: dType.SetJOGCmd(self.api, True, 1 if d > 0 else 2, isQueued=1)
    def _start_jog_reach(self, d):
        if self.is_connected: dType.SetJOGCmd(self.api, False, 1 if d > 0 else 2, isQueued=1)
    def _start_jog_raw(self, cid):
        if self.is_connected: dType.SetJOGCmd(self.api, False, cid, isQueued=1)
    def _stop_jog(self):
        if self.is_connected: dType.SetJOGCmd(self.api, False, 0, isQueued=1)

    def connect(self):
        if not DOBOT_AVAILABLE:
            self.log("DobotDllType がありません", "ERROR"); return
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
            self.log("Dobot接続完了")
            self._update_pos()
            self._update_step_display()
        else:
            self.log(f"接続失敗 (code={r[0]})", "ERROR")

    def _clear_alarm(self):
        if self.is_connected:
            dType.ClearAllAlarmsState(self.api)
            self.conn_label.config(text="接続済み ✓", fg="green")

    def _update_pos(self):
        if not self.is_connected: return
        p = dType.GetPose(self.api)
        j1 = math.degrees(math.atan2(p[1], p[0]))
        reach = math.sqrt(p[0]**2 + p[1]**2)
        if self.floor_z is not None:
            rel = p[2] - self.floor_z
            self.pos_label.config(
                text=f"J1={j1:7.1f}°  Reach={reach:6.1f}mm\n"
                     f"Z ={p[2]:7.1f}mm  R    ={p[3]:7.1f}°\n"
                     f"床から: {rel:.1f}mm")
        else:
            self.pos_label.config(
                text=f"J1={j1:7.1f}°  Reach={reach:6.1f}mm\n"
                     f"Z ={p[2]:7.1f}mm  R    ={p[3]:7.1f}°")

    def _periodic_update(self):
        if self.is_connected: self._update_pos()
        self.root.after(500, self._periodic_update)

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
        self.log(f"グリッパー {'閉じ' if close else '開き'}")

    # ======================== カメラ ========================
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

    def _update_camera(self):
        if not self.is_camera_running: return
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.current_frame = frame.copy()
            display = frame.copy()

            # 記録済みの点を表示
            if self.point1:
                px, py = self.point1[0], self.point1[1]
                cv2.drawMarker(display, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"P1(floor)", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self.point2:
                px, py = self.point2[0], self.point2[1]
                cv2.drawMarker(display, (px, py), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(display, f"P2(+{self.Z_UP_FROM_FLOOR:.0f})", (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if self.point1 and self.point2:
                cv2.line(display,
                         (self.point1[0], self.point1[1]),
                         (self.point2[0], self.point2[1]),
                         (0, 255, 255), 1, cv2.LINE_AA)

            # クリック待ち表示
            if self.recording_point == "point1":
                cv2.rectangle(display, (0, 0), (639, 479), (0, 0, 255), 4)
                cv2.putText(display, "Step 2: CLICK TARGET MARK", (130, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.recording_point == "point2":
                cv2.rectangle(display, (0, 0), (639, 479), (255, 0, 0), 4)
                cv2.putText(display, "Step 3: CLICK SAME MARK", (140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 十字線
            h, w = display.shape[:2]
            cv2.line(display, (w//2, 0), (w//2, h), (50, 50, 50), 1)
            cv2.line(display, (0, h//2), (w, h//2), (50, 50, 50), 1)

            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        self.root.after(33, self._update_camera)

    # ======================== ログ・終了 ========================
    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {msg}")
        self.root.after(0, lambda: (
            self.log_text.insert(tk.END, f"[{ts}] [{level}] {msg}\n"),
            self.log_text.see(tk.END)
        ))

    def on_closing(self):
        self.stop_camera()
        if self.is_connected and DOBOT_AVAILABLE:
            try: dType.DisconnectDobot(self.api)
            except: pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x750")
    app = TgtCalibrationTool(root)
    root.mainloop()