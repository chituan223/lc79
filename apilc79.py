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

app = Flask(__name__)

# ===============================
# LỊCH SỬ KHÔNG GIỚI HẠN
# ===============================
history   = []   
hist_pt   = []   
hist_dice = [] 

# ===============================
# THỐNG KÊ DỰ ĐOÁN
# ===============================
stats = {
    "tong":     0,
    "dung":     0,
    "sai":      0,
    "cd":       0,   # chuỗi đúng hiện tại
    "cs":       0,   # chuỗi sai hiện tại
    "max_cd":   0,
    "max_cs":   0,
}
_prev_pred  = None
_prev_phien = None

# ══════════════════════════════════════════════════════
#  AI ENGINE – 7 MÔ HÌNH
# ══════════════════════════════════════════════════════

# Bảng học
_t1  = defaultdict(lambda: {"Tài":0,"Xỉu":0})   # Markov bậc 1
_t2  = defaultdict(lambda: {"Tài":0,"Xỉu":0})   # Markov bậc 2
_t3  = defaultdict(lambda: {"Tài":0,"Xỉu":0})   # Markov bậc 3
_t4  = defaultdict(lambda: {"Tài":0,"Xỉu":0})   # Markov bậc 4
_ng  = defaultdict(lambda: {"Tài":0,"Xỉu":0})   # N-gram
_sd  = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}
_acc = {k: {"ok":0,"n":0} for k in ("m1","m2","m3","m4","ng","sk","pt","freq","mom")}
_prev_model = {}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 1-4: MARKOV CHAIN bậc 1→4
# Học xác suất P(kết quả | N kết quả trước)
# Bậc cao hơn → nhớ ngữ cảnh dài hơn
# ─────────────────────────────────────────────────────
def _train_markov():
    for d in _t1.values(): d.update({"Tài":0,"Xỉu":0})
    for d in _t2.values(): d.update({"Tài":0,"Xỉu":0})
    for d in _t3.values(): d.update({"Tài":0,"Xỉu":0})
    for d in _t4.values(): d.update({"Tài":0,"Xỉu":0})
    h=history
    for i in range(len(h)-1):
        _t1[h[i]][h[i+1]] += 1
    for i in range(len(h)-2):
        _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(len(h)-3):
        _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1
    for i in range(len(h)-4):
        _t4[h[i]+"|"+h[i+1]+"|"+h[i+2]+"|"+h[i+3]][h[i+4]] += 1

def _sc_markov():
    def _s(table, key):
        d=table.get(key,{"Tài":0,"Xỉu":0}); t=d["Tài"]+d["Xỉu"]
        if not t: return {"Tài":0.0,"Xỉu":0.0}
        return {"Tài":d["Tài"]/t,"Xỉu":d["Xỉu"]/t}
    h=history
    s1=_s(_t1, h[-1]) if len(h)>=1 else {"Tài":0.0,"Xỉu":0.0}
    s2=_s(_t2, h[-2]+"|"+h[-1]) if len(h)>=2 else {"Tài":0.0,"Xỉu":0.0}
    s3=_s(_t3, h[-3]+"|"+h[-2]+"|"+h[-1]) if len(h)>=3 else {"Tài":0.0,"Xỉu":0.0}
    s4=_s(_t4, h[-4]+"|"+h[-3]+"|"+h[-2]+"|"+h[-1]) if len(h)>=4 else {"Tài":0.0,"Xỉu":0.0}
    return s1,s2,s3,s4


# ─────────────────────────────────────────────────────
# MÔ HÌNH 5: N-GRAM tối đa 10 phiên
# Nhớ chuỗi lặp lại, pattern dài → trọng số bình phương
# ─────────────────────────────────────────────────────
def _train_ngram():
    _ng.clear()
    h=history
    for ln in range(1,11):
        for i in range(len(h)-ln):
            pat="|".join(h[i:i+ln])
            _ng[pat][h[i+ln]] += 1

def _sc_ngram():
    sc={"Tài":0.0,"Xỉu":0.0}
    h=history
    for ln in range(min(10,len(h)),0,-1):
        pat="|".join(h[-ln:])
        d=_ng.get(pat)
        if not d: continue
        t=d["Tài"]+d["Xỉu"]
        if not t: continue
        w=ln*ln*ln    # bậc 3 để ưu tiên pattern dài hơn nữa
        sc["Tài"]+=w*d["Tài"]/t
        sc["Xỉu"]+=w*d["Xỉu"]/t
    return sc


# ─────────────────────────────────────────────────────
# MÔ HÌNH 6: STREAK REVERSAL
# Học phân phối độ dài chuỗi → dự đoán đảo chiều
# ─────────────────────────────────────────────────────
def _train_streak():
    for d in _sd.values(): d.clear()
    if not history: return
    cur,cnt=history[0],1
    for r in history[1:]:
        if r==cur: cnt+=1
        else: _sd[cur][cnt]+=1; cur,cnt=r,1
    _sd[cur][cnt]+=1

def _cur_streak():
    if not history: return None,0
    cur=history[-1]; cnt=1
    for r in reversed(history[:-1]):
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


# ─────────────────────────────────────────────────────
# MÔ HÌNH 7: POINT BIAS
# Xu hướng tổng điểm xúc xắc gần đây
# ─────────────────────────────────────────────────────
def _sc_point(w=20):
    if len(hist_pt)<5: return {"Tài":0.5,"Xỉu":0.5}
    avg=sum(hist_pt[-w:])/len(hist_pt[-w:])
    p_t=max(0.0,min(1.0,(avg-3)/15))
    return {"Tài":p_t,"Xỉu":1-p_t}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 8: FREQUENCY WINDOW
# Tần suất Tài/Xỉu trong cửa sổ ngắn (10 phiên gần nhất)
# Khi lệch nhiều → dự đoán ngược lại (hồi quy trung bình)
# ─────────────────────────────────────────────────────
def _sc_freq(w=10):
    if len(history)<w: return {"Tài":0.5,"Xỉu":0.5}
    recent=history[-w:]
    ct=recent.count("Tài"); cx=w-ct
    # Nếu Tài nhiều → xu hướng Xỉu (hồi quy), và ngược lại
    p_xiu=ct/w   # càng nhiều Tài → p_xiu cao
    return {"Tài":1-p_xiu,"Xỉu":p_xiu}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 9: MOMENTUM
# Phân tích momentum ngắn hạn (5 phiên) vs dài hạn (20 phiên)
# Khi xu hướng ngắn và dài đồng thuận → tín hiệu mạnh
# ─────────────────────────────────────────────────────
def _sc_momentum():
    if len(history)<20: return {"Tài":0.5,"Xỉu":0.5}
    short=history[-5:];  ps=short.count("Tài")/5
    long =history[-20:]; pl=long.count("Tài")/20
    # Momentum = xu hướng ngắn hạn
    # Nếu ngắn > dài → đang tăng Tài
    diff=ps-pl   # [-1, 1]
    p_t=max(0.0,min(1.0,0.5+diff))
    return {"Tài":p_t,"Xỉu":1-p_t}


# ─────────────────────────────────────────────────────
# ENTROPY – đo độ hỗn loạn (entropy cao → giảm tin cậy)
# ─────────────────────────────────────────────────────
def _entropy(w=30):
    r=history[-w:]; n=len(r)
    if n==0: return 1.0
    ct=r.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))


# ─────────────────────────────────────────────────────
# ADAPTIVE WEIGHT – mô hình đúng nhiều → trọng số tăng
# ─────────────────────────────────────────────────────
def _aw(key, base):
    a=_acc[key]
    if a["n"]<20: return base
    rate=a["ok"]/a["n"]
    # Điều chỉnh ±50% dựa trên performance
    return max(0.01, base*(1+3.0*(rate-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_model_acc(actual):
    for k,p in _prev_model.items():
        if p:
            _acc[k]["n"]+=1
            if p==actual: _acc[k]["ok"]+=1

def _acc_str(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"


# ─────────────────────────────────────────────────────
# CẬP NHẬT THỐNG KÊ THẮNG / THUA
# ─────────────────────────────────────────────────────
def _update_stats(actual):
    global _prev_pred
    if not _prev_pred or _prev_pred=="Đang chờ":
        return
    stats["tong"] += 1
    if _prev_pred==actual:
        stats["dung"]+=1; stats["cd"]+=1; stats["cs"]=0
        if stats["cd"]>stats["max_cd"]: stats["max_cd"]=stats["cd"]
    else:
        stats["sai"]+=1; stats["cs"]+=1; stats["cd"]=0
        if stats["cs"]>stats["max_cs"]: stats["max_cs"]=stats["cs"]


# ===============================
# HÀM DỰ ĐOÁN CHÍNH
# ===============================
def get_prediction():
    if len(history) < 10:
        return "Đang chờ", 0.0

    # Train tất cả mô hình
    _train_markov()
    _train_ngram()
    _train_streak()

    e           = _entropy()
    s1,s2,s3,s4 = _sc_markov()
    sng         = _sc_ngram()
    ssk         = _sc_streak()
    spt         = _sc_point()
    sfr         = _sc_freq()
    smom        = _sc_momentum()

    # Trọng số thích nghi – entropy cao → giảm pattern
    e_factor = 1 - e*0.35

    w1   = _aw("m1",  0.08)
    w2   = _aw("m2",  0.12)
    w3   = _aw("m3",  0.15)
    w4   = _aw("m4",  0.15)
    wng  = _aw("ng",  0.22 * e_factor)
    wsk  = _aw("sk",  0.12)
    wpt  = _aw("pt",  0.07)
    wfr  = _aw("freq",0.05)
    wmom = _aw("mom", 0.04)
    tw   = w1+w2+w3+w4+wng+wsk+wpt+wfr+wmom

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(
            w1*s1.get(r,0)   + w2*s2.get(r,0)   +
            w3*s3.get(r,0)   + w4*s4.get(r,0)   +
            wng*sng.get(r,0) + wsk*ssk.get(r,0)  +
            wpt*spt.get(r,0) + wfr*sfr.get(r,0)  +
            wmom*smom.get(r,0)
        )/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred = "Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    conf = max(raw["Tài"],raw["Xỉu"])   # [0.5, 1.0]

    # Độ tin cậy THẬT = kết hợp:
    # 1. Accuracy lịch sử thực tế của ensemble
    # 2. Biên độ điểm (conf)
    # 3. Mức độ đồng thuận giữa các mô hình
    counted=[_acc[k]["ok"]/_acc[k]["n"] for k in _acc if _acc[k]["n"]>=10]
    hist_acc=sum(counted)/len(counted) if counted else 0.5

    preds=[_win(s1),_win(s2),_win(s3),_win(s4),
           _win(sng),_win(ssk),_win(spt),_win(sfr),_win(smom)]
    dong_thuan=preds.count(pred)/len(preds)   # tỷ lệ đồng thuận

    tin_cay_real=(
        hist_acc*0.50 +
        (conf-0.5)*2*0.30 +
        dong_thuan*0.20
    )
    # Chuẩn hoá về [50, 95]
    tin_cay_pct=round(50 + tin_cay_real*45, 1)
    tin_cay_pct=max(50.0, min(95.0, tin_cay_pct))

    global _prev_model
    _prev_model={
        "m1":_win(s1),"m2":_win(s2),"m3":_win(s3),"m4":_win(s4),
        "ng":_win(sng),"sk":_win(ssk),"pt":_win(spt),
        "freq":_win(sfr),"mom":_win(smom),
    }

    return pred, tin_cay_pct


# ===============================
# BOT NỀN – LẤY DATA 24/7
# ===============================
latest_data = {}
last_id     = None

def fetch_data_loop():
    global latest_data, last_id, _prev_pred

    while True:
        try:
            res  = requests.get(API_URL, timeout=10)
            data = res.json()
            lst  = data.get("list", [])
            if not lst: time.sleep(2); continue

            phien    = lst[0]
            phien_id = phien.get("id")
            if phien_id == last_id: time.sleep(2); continue

            dices = phien.get("dices")
            tong  = phien.get("point")
            d1,d2,d3 = dices
            ket   = "Tài" if tong>=11 else "Xỉu"

            # Cập nhật accuracy & stats với kết quả thực
            _update_stats(ket)
            if len(history)>=10:
                _update_model_acc(ket)

            # Lưu lịch sử
            history.append(ket)
            hist_pt.append(tong)
            hist_dice.append((d1,d2,d3))

            # Dự đoán phiên TIẾP THEO
            pred, tin_cay = get_prediction()
            _prev_pred = pred
            last_id    = phien_id

            phien_tiep = phien_id + 1

            # Thống kê tỷ lệ
            tong_dd  = stats["tong"]
            acc_tong = f"{stats['dung']}/{tong_dd} ({stats['dung']/tong_dd*100:.1f}%)" if tong_dd else "Chưa có"
            so_phien = len(history)
            cur_val,cur_len=_cur_streak()

           
            latest_data = {
                "Phiên":          phien_id,
                "Xúc xắc 1":      d1,
                "Xúc xắc 2":      d2,
                "Xúc xắc 3":      d3,
                "Tổng":           tong,
                "Kết":            ket,
                "phien_hien_tai": phien_tiep,
                "Dự đoán":        pred,
                "Độ tin cậy":     f"{tin_cay}%",
                "ID":             "tuananh"
            }

          
            
            print(f"  Phiên         : {phien_id}")
            print(f"  Xúc xắc      : {d1}  {d2}  {d3}")
            print(f"  Tổng          : {tong}")
            print(f"  Kết quả       : {ket}")
            print(f"  Chuỗi         : {cur_val} x{cur_len}" if cur_val else "  Chuỗi         : --")
            print(f"  Số phiên học  : {so_phien}")
            

            if pred=="Đang chờ":
                print(f"  Dự đoán       : Chờ thêm {10-so_phien} phiên...")
            else:
                print(f"  Dự đoán P.{phien_tiep:<6}: >>> {pred} <<<")
                print(f"  Độ tin cậy    : {tin_cay}%")
                
                print("  Thống kê dự đoán:")
                print(f"    Tổng đã đoán : {tong_dd}")
                print(f"    Đúng / Sai   : {stats['dung']} / {stats['sai']}")
                print(f"    Tỷ lệ đúng   : {acc_tong}")
                print(f"    Chuỗi đúng   : {stats['cd']} (max {stats['max_cd']})")
                print(f"    Chuỗi sai    : {stats['cs']} (max {stats['max_cs']})")
                
                print("  📈 Thống kê độ chuẩn mô hình:")
                
                for lbl,key in [
                    ("Markov 1 ","m1"),("Markov 2 ","m2"),
                    ("Markov 3 ","m3"),("Markov 4 ","m4"),
                    ("N-Gram   ","ng"),("Streak   ","sk"),
                    ("PointBias","pt"),("Frequency","freq"),
                    ("Momentum ","mom"),
                ]:
                    print(f"    {lbl}: {_acc_str(key)}")

                   
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi:", e)

        time.sleep(2)


# ===============================
# KHỞI CHẠY
# ===============================
threading.Thread(target=fetch_data_loop, daemon=True).start()


# ===============================
# API
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    if latest_data:
        return jsonify({"data": latest_data})
    return jsonify({"status": "Đang khởi động..."})


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    
