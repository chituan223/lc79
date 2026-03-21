import requests
import time
import threading
import math
from flask import Flask, jsonify
from datetime import datetime
from collections import defaultdict

# ===============================
# CẤU HÌNH
# ===============================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
last_processed_session_id = None

app = Flask(__name__)

# ===============================
# DỮ LIỆU TRẢ VỀ API
# ===============================
latest_data = {
    "Phiên":          None,
    "Xúc xắc 1":      None,
    "Xúc xắc 2":      None,
    "Xúc xắc 3":      None,
    "Tổng":           None,
    "Kết":            None,
    "Phiên hiện tại": None,
    "Dự đoán":        "Đang chờ",
    "Độ tin cậy":     0.0,  # %
    "ID":             "tuananh"
}

# ===============================
# LỊCH SỬ KHÔNG GIỚI HẠN
# ===============================
history_all   = []   # "Tài" / "Xỉu"
history_point = []   # tổng điểm thực

# ── Thống kê thắng / thua dự đoán ──
stats = {
    "tong_phien":    0,
    "du_doan_dung":  0,
    "du_doan_sai":   0,
    "chuoi_dung":    0,
    "chuoi_sai":     0,
    "max_chuoi_dung": 0,
    "max_chuoi_sai":  0,
}
_prev_pred = None

# ══════════════════════════════════════════════
#  AI ENGINE – giữ nguyên code gốc
# ══════════════════════════════════════════════

_t1 = defaultdict(lambda: {"Tài": 0, "Xỉu": 0})
_t2 = defaultdict(lambda: {"Tài": 0, "Xỉu": 0})
_t3 = defaultdict(lambda: {"Tài": 0, "Xỉu": 0})
_ng = defaultdict(lambda: {"Tài": 0, "Xỉu": 0})
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}
_acc = {k: {"ok": 0, "n": 0} for k in ("m1","m2","m3","ng","sk","pt")}
_prev_model = {}

# --- MARKOV ---
def _train_markov():
    for d in _t1.values(): d.update({"Tài":0,"Xỉu":0})
    for d in _t2.values(): d.update({"Tài":0,"Xỉu":0})
    for d in _t3.values(): d.update({"Tài":0,"Xỉu":0})
    h = history_all
    for i in range(len(h)-1):
        _t1[h[i]][h[i+1]] += 1
    for i in range(len(h)-2):
        _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(len(h)-3):
        _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1

def _sc_markov():
    s1={"Tài":0.0,"Xỉu":0.0}
    s2={"Tài":0.0,"Xỉu":0.0}
    s3={"Tài":0.0,"Xỉu":0.0}
    h=history_all
    if len(h)>=1:
        d=_t1[h[-1]]; t=d["Tài"]+d["Xỉu"]
        if t: s1["Tài"]=d["Tài"]/t; s1["Xỉu"]=d["Xỉu"]/t
    if len(h)>=2:
        d=_t2.get(h[-2]+"|"+h[-1],{"Tài":0,"Xỉu":0}); t=d["Tài"]+d["Xỉu"]
        if t: s2["Tài"]=d["Tài"]/t; s2["Xỉu"]=d["Xỉu"]/t
    if len(h)>=3:
        d=_t3.get(h[-3]+"|"+h[-2]+"|"+h[-1],{"Tài":0,"Xỉu":0}); t=d["Tài"]+d["Xỉu"]
        if t: s3["Tài"]=d["Tài"]/t; s3["Xỉu"]=d["Xỉu"]/t
    return s1,s2,s3

# --- N-GRAM ---
def _train_ngram():
    _ng.clear()
    h=history_all
    for ln in range(1,9):
        for i in range(len(h)-ln):
            pat="|".join(h[i:i+ln])
            _ng[pat][h[i+ln]] += 1

def _sc_ngram():
    sc={"Tài":0.0,"Xỉu":0.0}
    h=history_all
    for ln in range(min(8,len(h)),0,-1):
        pat="|".join(h[-ln:])
        d=_ng.get(pat)
        if not d: continue
        t=d["Tài"]+d["Xỉu"]
        if not t: continue
        w=ln*ln
        sc["Tài"]+=w*d["Tài"]/t
        sc["Xỉu"]+=w*d["Xỉu"]/t
    return sc

# --- Streak ---
def _train_streak():
    for d in _sd.values(): d.clear()
    if not history_all: return
    cur,cnt=history_all[0],1
    for r in history_all[1:]:
        if r==cur: cnt+=1
        else: _sd[cur][cnt]+=1; cur,cnt=r,1
    _sd[cur][cnt]+=1

def _cur_streak():
    if not history_all: return None,0
    cur=history_all[-1]; cnt=1
    for r in reversed(history_all[:-1]):
        if r==cur: cnt+=1
        else: break
    return cur,cnt

def _sc_streak():
    cur,ln=_cur_streak()
    if not cur: return {"Tài":0.5,"Xỉu":0.5}
    dist=_sd[cur]
    ended =sum(v for k,v in dist.items() if k<=ln)
    longer=sum(v for k,v in dist.items() if k>ln)
    total =ended+longer
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    other="Xỉu" if cur=="Tài" else "Tài"
    return {cur:longer/total, other:ended/total}

# --- Point Bias ---
def _sc_point(w=20):
    if len(history_point)<5: return {"Tài":0.5,"Xỉu":0.5}
    recent=history_point[-w:]
    avg=sum(recent)/len(recent)
    p_t=max(0.0,min(1.0,(avg-3)/15))
    return {"Tài":p_t,"Xỉu":1-p_t}

# --- Entropy ---
def _entropy(w=30):
    r=history_all[-w:]; n=len(r)
    if n==0: return 1.0
    ct=r.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))

# --- Adaptive weight ---
def _aw(key,base):
    a=_acc[key]
    if a["n"]<15: return base
    return max(0.02, base*(1+2.5*(a["ok"]/a["n"]-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_model_acc(actual):
    for k,pred in _prev_model.items():
        if pred:
            _acc[k]["n"]+=1
            if pred==actual: _acc[k]["ok"]+=1

def _acc_pct(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"

# --- Cập nhật thống kê ---
def _update_stats(actual):
    global _prev_pred
    if _prev_pred is None or _prev_pred == "Đang chờ":
        return
    stats["tong_phien"] += 1
    if _prev_pred == actual:
        stats["du_doan_dung"]  += 1
        stats["chuoi_dung"]    += 1
        stats["chuoi_sai"]      = 0
        if stats["chuoi_dung"] > stats["max_chuoi_dung"]:
            stats["max_chuoi_dung"] = stats["chuoi_dung"]
    else:
        stats["du_doan_sai"]   += 1
        stats["chuoi_sai"]     += 1
        stats["chuoi_dung"]     = 0
        if stats["chuoi_sai"] > stats["max_chuoi_sai"]:
            stats["max_chuoi_sai"] = stats["chuoi_sai"]

# ===============================
# HÀM DỰ ĐOÁN CHÍNH – trả pred + confidence %
# ===============================
def get_prediction():
    if len(history_all) < 10:
        return "Đang chờ", 0.0

    _train_markov()
    _train_ngram()
    _train_streak()

    e        = _entropy()
    s1,s2,s3 = _sc_markov()
    sng      = _sc_ngram()
    ssk      = _sc_streak()
    spt      = _sc_point()

    w1  = _aw("m1", 0.10)
    w2  = _aw("m2", 0.15)
    w3  = _aw("m3", 0.20)
    wng = _aw("ng", 0.30*(1-e*0.3))
    wsk = _aw("sk", 0.15)
    wpt = _aw("pt", 0.10)
    tw  = w1+w2+w3+wng+wsk+wpt

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(
            w1*s1.get(r,0)+w2*s2.get(r,0)+w3*s3.get(r,0)
            +wng*sng.get(r,0)+wsk*ssk.get(r,0)+wpt*spt.get(r,0)
        )/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred="Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    confidence = max(raw["Tài"], raw["Xỉu"])*100  # độ tin cậy %

    global _prev_model
    _prev_model={
        "m1":_win(s1),"m2":_win(s2),"m3":_win(s3),
        "ng":_win(sng),"sk":_win(ssk),"pt":_win(spt),
    }

    return pred, confidence


# ===============================
# BOT NỀN
# ===============================
def fetch_data_loop():
    global last_processed_session_id, latest_data, _prev_pred

    while True:
        try:
            res      = requests.get(API_URL, timeout=10)
            data     = res.json()
            list_data= data.get("list", [])
            if not list_data:
                time.sleep(2)
                continue

            phien    = list_data[0]
            phien_id = phien.get("id")

            if phien_id == last_processed_session_id:
                time.sleep(2)
                continue

            dices = phien.get("dices")
            tong  = phien.get("point")
            d1,d2,d3 = dices
            ket_qua  = "Tài" if tong >= 11 else "Xỉu"

            # Cập nhật thống kê
            _update_stats(ket_qua)
            if len(history_all) >= 10:
                _update_model_acc(ket_qua)

            # Lưu lịch sử
            history_all.append(ket_qua)
            history_point.append(tong)

            # Dự đoán phiên tiếp theo
            prediction, confidence = get_prediction()
            _prev_pred = prediction

            last_processed_session_id = phien_id

            # Cập nhật JSON
            latest_data.update({
                "Phiên":          phien_id,
                "Xúc xắc 1":      d1,
                "Xúc xắc 2":      d2,
                "Xúc xắc 3":      d3,
                "Tổng":           tong,
                "Kết":            ket_qua,
                "Phiên hiện tại": phien_id,
                "Dự đoán":        prediction,
                "Độ tin cậy":     round(confidence,1),
                "ID":             "tuananh"
            })

        except Exception as e:
            print(f"Lỗi ({datetime.now().strftime('%H:%M:%S')}):", e)

        time.sleep(2)

# ===============================
# KHỞI CHẠY TIẾN TRÌNH NỀN
# ===============================
threading.Thread(target=fetch_data_loop, daemon=True).start()

# ===============================
# API CHÍNH
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify({"data": latest_data})

# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    print("Server đang chạy…")
    app.run(host="0.0.0.0", port=10000)