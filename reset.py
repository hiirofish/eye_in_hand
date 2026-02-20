import DobotDllType as dType

api = dType.load()
# 接続
state = dType.ConnectDobot(api, "COM3", 115200)[0]

if state == 0:
    print("接続しました。アラームをクリアします...")
    # ★これがアラーム解除コマンドです
    dType.ClearAllAlarmsState(api)
    print("完了。緑ランプになりましたか？")
    dType.DisconnectDobot(api)
else:
    print("接続できません")