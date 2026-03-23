import requests
import time
import threading
import math
from flask import Flask, jsonify
from datetime import datetime
from collections import defaultdict, deque

# ===============================
# CẤU HÌNH
# ===============================
API_URL    = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MIN_PHIEN  = 20    # bắt đầu dự đoán sau 20 phiên
MAX_PHIEN  = 200   # giữ tối đa 200 phiên

app = Flask(__name__)

# ===============================
# LỊCH SỬ – tối đa 200 phiên
# ===============================
history   = deque(maxlen=MAX_PHIEN)   # "Tài"/"Xỉu"
hist_pt   = deque(maxlen=MAX_PHIEN)   # tổng điểm
hist_dice = deque(maxlen=MAX_PHIEN)   # (d1,d2,d3)

# ===============================
# THỐNG KÊ DỰ ĐOÁN
# ===============================
stats = {
    "tong":0,"dung":0,"sai":0,
    "cd":0,"cs":0,"max_cd":0,"max_cs":0,
}
_prev_pred = None

# ══════════════════════════════════════════════════════
#  AI ENGINE – 12 MÔ HÌNH
# ══════════════════════════════════════════════════════

_t1 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t2 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t3 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t4 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t5 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_ng = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}

_acc = {k: {"ok":0,"n":0} for k in (
    "m1","m2","m3","m4","m5",
    "ng","sk","pt","fr10","fr20",
    "mom","rep"
)}
_prev_model = {}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 1-5: MARKOV CHAIN bậc 1→5
# P(kết quả tiếp | N kết quả trước)
# ─────────────────────────────────────────────────────
def _train_markov():
    for tb in (_t1,_t2,_t3,_t4,_t5):
        for d in tb.values(): d.update({"Tài":0,"Xỉu":0})
    h = list(history)
    n = len(h)
    for i in range(n-1): _t1[h[i]][h[i+1]] += 1
    for i in range(n-2): _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(n-3): _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1
    for i in range(n-4): _t4[h[i]+"|"+h[i+1]+"|"+h[i+2]+"|"+h[i+3]][h[i+4]] += 1
    for i in range(n-5): _t5["|".join(h[i:i+5])][h[i+5]] += 1

def _s(table, key):
    d = table.get(key, {"Tài":0,"Xỉu":0})
    t = d["Tài"]+d["Xỉu"]
    if not t: return {"Tài":0.0,"Xỉu":0.0}
    return {"Tài":d["Tài"]/t,"Xỉu":d["Xỉu"]/t}

def _sc_markov():
    h = list(history)
    s1 = _s(_t1, h[-1])                        if len(h)>=1 else {"Tài":0.0,"Xỉu":0.0}
    s2 = _s(_t2, h[-2]+"|"+h[-1])              if len(h)>=2 else {"Tài":0.0,"Xỉu":0.0}
    s3 = _s(_t3, "|".join(h[-3:]))             if len(h)>=3 else {"Tài":0.0,"Xỉu":0.0}
    s4 = _s(_t4, "|".join(h[-4:]))             if len(h)>=4 else {"Tài":0.0,"Xỉu":0.0}
    s5 = _s(_t5, "|".join(h[-5:]))             if len(h)>=5 else {"Tài":0.0,"Xỉu":0.0}
    return s1,s2,s3,s4,s5


# ─────────────────────────────────────────────────────
# MÔ HÌNH 6: N-GRAM tối đa 12 phiên
# Pattern dài → trọng số bậc 4
# ─────────────────────────────────────────────────────
def _train_ngram():
    _ng.clear()
    h = list(history)
    for ln in range(1, 13):
        for i in range(len(h)-ln):
            pat = "|".join(h[i:i+ln])
            _ng[pat][h[i+ln]] += 1

def _sc_ngram():
    sc = {"Tài":0.0,"Xỉu":0.0}
    h  = list(history)
    for ln in range(min(12,len(h)),0,-1):
        pat = "|".join(h[-ln:])
        d   = _ng.get(pat)
        if not d: continue
        t = d["Tài"]+d["Xỉu"]
        if not t: continue
        w = ln**4    # bậc 4 → ưu tiên mạnh pattern dài
        sc["Tài"] += w*d["Tài"]/t
        sc["Xỉu"] += w*d["Xỉu"]/t
    return sc


# ─────────────────────────────────────────────────────
# MÔ HÌNH 7: STREAK REVERSAL sâu hơn
# Phân tích phân phối chuỗi có trọng số theo độ dài
# ─────────────────────────────────────────────────────
def _train_streak():
    for d in _sd.values(): d.clear()
    h = list(history)
    if not h: return
    cur,cnt = h[0],1
    for r in h[1:]:
        if r==cur: cnt+=1
        else: _sd[cur][cnt]+=1; cur,cnt=r,1
    _sd[cur][cnt]+=1

def _cur_streak():
    h = list(history)
    if not h: return None,0
    cur=h[-1]; cnt=1
    for r in reversed(h[:-1]):
        if r==cur: cnt+=1
        else: break
    return cur,cnt

def _sc_streak():
    cur,ln = _cur_streak()
    if not cur: return {"Tài":0.5,"Xỉu":0.5}
    dist  = _sd[cur]
    # Weighted: chuỗi dài hơn có trọng số cao hơn
    ended  = sum(v*(k**1.5) for k,v in dist.items() if k<=ln)
    longer = sum(v*(k**1.5) for k,v in dist.items() if k>ln)
    total  = ended+longer
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    other="Xỉu" if cur=="Tài" else "Tài"
    return {cur:longer/total, other:ended/total}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 8: POINT BIAS – xu hướng tổng điểm
# Phân tích xu hướng tổng điểm xúc xắc thực tế
# ─────────────────────────────────────────────────────
def _sc_point(w=30):
    pts = list(hist_pt)
    if len(pts)<10: return {"Tài":0.5,"Xỉu":0.5}
    recent = pts[-w:]
    avg = sum(recent)/len(recent)
    # Thêm phân tích xu hướng (slope)
    n   = len(recent)
    if n>=6:
        half1=sum(recent[:n//2])/(n//2)
        half2=sum(recent[n//2:])/(n-n//2)
        slope=(half2-half1)/10.5   # normalized [-1,1]
    else:
        slope=0
    p_t = max(0.0, min(1.0, (avg-3)/15 + slope*0.1))
    return {"Tài":p_t,"Xỉu":1-p_t}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 9: FREQUENCY WINDOW 10 phiên
# Hồi quy trung bình ngắn hạn
# ─────────────────────────────────────────────────────
def _sc_freq10():
    h = list(history)
    w = 10
    if len(h)<w: return {"Tài":0.5,"Xỉu":0.5}
    ct  = h[-w:].count("Tài")
    p_x = ct/w
    return {"Tài":1-p_x,"Xỉu":p_x}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 10: FREQUENCY WINDOW 20 phiên
# Hồi quy trung bình trung hạn
# ─────────────────────────────────────────────────────
def _sc_freq20():
    h = list(history)
    w = 20
    if len(h)<w: return {"Tài":0.5,"Xỉu":0.5}
    ct  = h[-w:].count("Tài")
    p_x = ct/w
    return {"Tài":1-p_x,"Xỉu":p_x}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 11: MOMENTUM đa tầng
# So sánh 3 cửa sổ: 5 / 15 / 30 phiên
# ─────────────────────────────────────────────────────
def _sc_momentum():
    h = list(history)
    if len(h)<30: return {"Tài":0.5,"Xỉu":0.5}
    p5  = h[-5:].count("Tài")/5
    p15 = h[-15:].count("Tài")/15
    p30 = h[-30:].count("Tài")/30
    # Momentum ngắn vs trung vs dài
    mom = (p5-p15)*0.6 + (p15-p30)*0.4
    p_t = max(0.0, min(1.0, 0.5+mom))
    return {"Tài":p_t,"Xỉu":1-p_t}


# ─────────────────────────────────────────────────────
# MÔ HÌNH 12: REPETITION PATTERN
# Phát hiện chu kỳ lặp lại (2,3,4 phiên)
# VD: TXTXTX → chu kỳ 2 → dự đoán tiếp theo
# ─────────────────────────────────────────────────────
def _sc_repeat():
    h = list(history)
    if len(h)<8: return {"Tài":0.5,"Xỉu":0.5}
    sc = {"Tài":0.0,"Xỉu":0.0}
    for cycle in (2,3,4,5):
        if len(h)<cycle*3: continue
        # So sánh chuỗi cuối với chuỗi cycle phiên trước
        match=0
        for i in range(1,4):
            if h[-i] == h[-i-cycle]: match+=1
        if match>=2:    # khớp ít nhất 2/3
            pred = h[-cycle]   # lặp lại theo chu kỳ
            w    = match*cycle
            sc[pred] += w
    total=sc["Tài"]+sc["Xỉu"]
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    return {"Tài":sc["Tài"]/total,"Xỉu":sc["Xỉu"]/total}


# ─────────────────────────────────────────────────────
# ENTROPY – độ hỗn loạn dữ liệu
# ─────────────────────────────────────────────────────
def _entropy(w=40):
    h = list(history)[-w:]
    n = len(h)
    if n==0: return 1.0
    ct=h.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))


# ─────────────────────────────────────────────────────
# ADAPTIVE WEIGHT – tự điều chỉnh theo accuracy thực
# ─────────────────────────────────────────────────────
def _aw(key, base):
    a=_acc[key]
    if a["n"]<20: return base
    rate=a["ok"]/a["n"]
    return max(0.005, base*(1+4.0*(rate-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_model_acc(actual):
    for k,p in _prev_model.items():
        if p: _acc[k]["n"]+=1; _acc[k]["ok"]+=(p==actual)

def _acc_str(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"


# ─────────────────────────────────────────────────────
# CẬP NHẬT THỐNG KÊ
# ─────────────────────────────────────────────────────
def _update_stats(actual):
    global _prev_pred
    if not _prev_pred or _prev_pred=="Đang chờ": return
    stats["tong"]+=1
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
    if len(history) < MIN_PHIEN:
        return "Đang chờ", 0.0

    _train_markov()
    _train_ngram()
    _train_streak()

    e              = _entropy()
    s1,s2,s3,s4,s5 = _sc_markov()
    sng            = _sc_ngram()
    ssk            = _sc_streak()
    spt            = _sc_point()
    sf10           = _sc_freq10()
    sf20           = _sc_freq20()
    smom           = _sc_momentum()
    srep           = _sc_repeat()

    ef = max(0.3, 1 - e*0.4)   # entropy factor

    w1   = _aw("m1",  0.06)
    w2   = _aw("m2",  0.09)
    w3   = _aw("m3",  0.12)
    w4   = _aw("m4",  0.13)
    w5   = _aw("m5",  0.13)
    wng  = _aw("ng",  0.18*ef)
    wsk  = _aw("sk",  0.10)
    wpt  = _aw("pt",  0.06)
    wf10 = _aw("fr10",0.04)
    wf20 = _aw("fr20",0.04)
    wmom = _aw("mom", 0.03)
    wrep = _aw("rep", 0.02)
    tw   = w1+w2+w3+w4+w5+wng+wsk+wpt+wf10+wf20+wmom+wrep

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(
            w1*s1.get(r,0)    + w2*s2.get(r,0)   +
            w3*s3.get(r,0)    + w4*s4.get(r,0)   +
            w5*s5.get(r,0)    + wng*sng.get(r,0)  +
            wsk*ssk.get(r,0)  + wpt*spt.get(r,0)  +
            wf10*sf10.get(r,0)+ wf20*sf20.get(r,0)+
            wmom*smom.get(r,0)+ wrep*srep.get(r,0)
        )/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred = "Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    conf = max(raw["Tài"],raw["Xỉu"])

    # Độ tin cậy THẬT
    counted=[_acc[k]["ok"]/_acc[k]["n"] for k in _acc if _acc[k]["n"]>=15]
    hist_acc=sum(counted)/len(counted) if counted else 0.5

    all_preds=[_win(s1),_win(s2),_win(s3),_win(s4),_win(s5),
               _win(sng),_win(ssk),_win(spt),_win(sf10),
               _win(sf20),_win(smom),_win(srep)]
    dong_thuan=all_preds.count(pred)/len(all_preds)

    tin_cay=(
        hist_acc*0.45 +
        (conf-0.5)*2*0.35 +
        dong_thuan*0.20
    )
    tin_cay_pct=round(max(50.0, min(95.0, 50+tin_cay*45)), 1)

    global _prev_model
    _prev_model={
        "m1":_win(s1),"m2":_win(s2),"m3":_win(s3),
        "m4":_win(s4),"m5":_win(s5),"ng":_win(sng),
        "sk":_win(ssk),"pt":_win(spt),"fr10":_win(sf10),
        "fr20":_win(sf20),"mom":_win(smom),"rep":_win(srep),
    }

    return pred, tin_cay_pct


# ===============================
# BOT NỀN
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
            if phien_id==last_id: time.sleep(2); continue

            dices    = phien.get("dices")
            tong     = phien.get("point")
            d1,d2,d3 = dices
            ket      = "Tài" if tong>=11 else "Xỉu"

            _update_stats(ket)
            if len(history)>=MIN_PHIEN:
                _update_model_acc(ket)

            history.append(ket)
            hist_pt.append(tong)
            hist_dice.append((d1,d2,d3))

            pred, tin_cay = get_prediction()
            _prev_pred = pred
            last_id    = phien_id
            phien_tiep = phien_id+1
            so_phien   = len(history)
            cur_val,cur_len = _cur_streak()
            tong_dd  = stats["tong"]
            acc_tong = f"{stats['dung']}/{tong_dd} ({stats['dung']/tong_dd*100:.1f}%)" if tong_dd else "Chưa có"

            # ── JSON TRẢ VỀ ──────────────────────────────
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
            print(f"  Bộ nhớ        : {so_phien}/{MAX_PHIEN} phiên")
            

            if pred=="Đang chờ":
                con_lai=MIN_PHIEN-so_phien
                print(f"  Dự đoán       : Chờ thêm {con_lai} phiên...")
            else:
                print(f"  Dự đoán P.{phien_tiep:<7}: >>> {pred} <<<")
                print(f"  Độ tin cậy    : {tin_cay}%")
                
                print(f"  Thống kê dự đoán:")
                print(f"    Tổng đã đoán : {tong_dd} phiên")
                print(f"    Đúng / Sai   : {stats['dung']} / {stats['sai']}")
                print(f"    Tỷ lệ đúng   : {acc_tong}")
                print(f"    Chuỗi đúng   : {stats['cd']} (max {stats['max_cd']})")
                print(f"    Chuỗi sai    : {stats['cs']} (max {stats['max_cs']})")
                
                print("  Accuracy 12 mô hình:")
                for lbl,key in [
                    ("Markov 1 ","m1"), ("Markov 2 ","m2"),
                    ("Markov 3 ","m3"), ("Markov 4 ","m4"),
                    ("Markov 5 ","m5"), ("N-Gram   ","ng"),
                    ("Streak   ","sk"), ("PointBias","pt"),
                    ("Freq-10  ","fr10"),("Freq-20  ","fr20"),
                    ("Momentum ","mom"),("Repeat   ","rep"),
                ]:
                    print(f"    {lbl}: {_acc_str(key)}")

            print(f"  ID            : tuananh")
            

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
