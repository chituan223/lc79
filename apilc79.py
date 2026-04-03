import requests
import time
import threading
import math
from flask import Flask, jsonify
from collections import defaultdict, deque
from datetime import datetime

# ===============================
# CẤU HÌNH
# ===============================
API_URL    = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MIN_PHIEN  = 20
MAX_PHIEN  = 100

app = Flask(__name__)

# ===============================
# DỮ LIỆU (GIỮ NGUYÊN JSON GỐC)
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
    "Độ tin cậy":     0.0,
    "ID":             "tuananh"
}

last_processed_session_id = None

# ===============================
# LỊCH SỬ
# ===============================
history  = deque(maxlen=MAX_PHIEN)
hist_pt  = deque(maxlen=MAX_PHIEN)

# ===============================
# AI ENGINE
# ===============================

_t1 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t2 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t3 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t4 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t5 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_ng = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}
_acc = {k: {"ok":0,"n":0} for k in ("m1","m2","m3","m4","m5","ng","sk","pt","fr10","fr20","mom","rep")}
_prev_model = {}
_prev_pred  = None
stats = {"tong":0,"dung":0,"sai":0,"cd":0,"cs":0,"max_cd":0,"max_cs":0}


def _train_markov():
    for tb in (_t1,_t2,_t3,_t4,_t5):
        for d in tb.values(): d.update({"Tài":0,"Xỉu":0})
    h = list(history)
    for i in range(len(h)-1): _t1[h[i]][h[i+1]] += 1
    for i in range(len(h)-2): _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(len(h)-3): _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1
    for i in range(len(h)-4): _t4[h[i]+"|"+h[i+1]+"|"+h[i+2]+"|"+h[i+3]][h[i+4]] += 1
    for i in range(len(h)-5): _t5["|".join(h[i:i+5])][h[i+5]] += 1

def _s(table, key):
    d = table.get(key, {"Tài":0,"Xỉu":0}); t = d["Tài"]+d["Xỉu"]
    if not t: return {"Tài":0.0,"Xỉu":0.0}
    return {"Tài":d["Tài"]/t,"Xỉu":d["Xỉu"]/t}

def _sc_markov():
    h = list(history)
    s1 = _s(_t1, h[-1])            if len(h)>=1 else {"Tài":0.0,"Xỉu":0.0}
    s2 = _s(_t2, h[-2]+"|"+h[-1])  if len(h)>=2 else {"Tài":0.0,"Xỉu":0.0}
    s3 = _s(_t3, "|".join(h[-3:])) if len(h)>=3 else {"Tài":0.0,"Xỉu":0.0}
    s4 = _s(_t4, "|".join(h[-4:])) if len(h)>=4 else {"Tài":0.0,"Xỉu":0.0}
    s5 = _s(_t5, "|".join(h[-5:])) if len(h)>=5 else {"Tài":0.0,"Xỉu":0.0}
    return s1,s2,s3,s4,s5

def _train_ngram():
    _ng.clear()
    h = list(history)
    for ln in range(1,13):
        for i in range(len(h)-ln):
            _ng["|".join(h[i:i+ln])][h[i+ln]] += 1

def _sc_ngram():
    sc={"Tài":0.0,"Xỉu":0.0}; h=list(history)
    for ln in range(min(12,len(h)),0,-1):
        pat="|".join(h[-ln:]); d=_ng.get(pat)
        if not d: continue
        t=d["Tài"]+d["Xỉu"]
        if not t: continue
        w=ln**4; sc["Tài"]+=w*d["Tài"]/t; sc["Xỉu"]+=w*d["Xỉu"]/t
    return sc

def _train_streak():
    for d in _sd.values(): d.clear()
    h=list(history)
    if not h: return
    cur,cnt=h[0],1
    for r in h[1:]:
        if r==cur: cnt+=1
        else: _sd[cur][cnt]+=1; cur,cnt=r,1
    _sd[cur][cnt]+=1

def _cur_streak():
    h=list(history)
    if not h: return None,0
    cur=h[-1]; cnt=1
    for r in reversed(h[:-1]):
        if r==cur: cnt+=1
        else: break
    return cur,cnt

def _sc_streak():
    cur,ln=_cur_streak()
    if not cur: return {"Tài":0.5,"Xỉu":0.5}
    dist=_sd[cur]
    ended =sum(v*(k**1.5) for k,v in dist.items() if k<=ln)
    longer=sum(v*(k**1.5) for k,v in dist.items() if k>ln)
    total =ended+longer
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    other="Xỉu" if cur=="Tài" else "Tài"
    return {cur:longer/total, other:ended/total}

def _sc_point(w=20):
    pts=list(hist_pt)
    if len(pts)<5: return {"Tài":0.5,"Xỉu":0.5}
    avg=sum(pts[-w:])/len(pts[-w:])
    p_t=max(0.0,min(1.0,(avg-3)/15))
    return {"Tài":p_t,"Xỉu":1-p_t}

def _sc_freq(w):
    h=list(history)
    if len(h)<w: return {"Tài":0.5,"Xỉu":0.5}
    ct=h[-w:].count("Tài"); p_x=ct/w
    return {"Tài":1-p_x,"Xỉu":p_x}

def _sc_momentum():
    h=list(history)
    if len(h)<30: return {"Tài":0.5,"Xỉu":0.5}
    p5=h[-5:].count("Tài")/5; p15=h[-15:].count("Tài")/15; p30=h[-30:].count("Tài")/30
    mom=(p5-p15)*0.6+(p15-p30)*0.4
    return {"Tài":max(0.0,min(1.0,0.5+mom)),"Xỉu":max(0.0,min(1.0,0.5-mom))}

def _sc_repeat():
    h=list(history)
    if len(h)<8: return {"Tài":0.5,"Xỉu":0.5}
    sc={"Tài":0.0,"Xỉu":0.0}
    for cycle in (2,3,4,5):
        if len(h)<cycle*3: continue
        match=sum(1 for i in range(1,4) if h[-i]==h[-i-cycle])
        if match>=2:
            w=match*cycle; sc[h[-cycle]]+=w
    total=sc["Tài"]+sc["Xỉu"]
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    return {"Tài":sc["Tài"]/total,"Xỉu":sc["Xỉu"]/total}

def _entropy(w=40):
    h=list(history)[-w:]; n=len(h)
    if n==0: return 1.0
    ct=h.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))

def _aw(key, base):
    a=_acc[key]
    if a["n"]<20: return base
    return max(0.005, base*(1+4.0*(a["ok"]/a["n"]-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_acc(actual):
    for k,p in _prev_model.items():
        if p: _acc[k]["n"]+=1; _acc[k]["ok"]+=(p==actual)

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

def _acc_str(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"

def get_prediction():
    if len(history)<MIN_PHIEN: return "Đang chờ", 0.0

    _train_markov(); _train_ngram(); _train_streak()
    e=_entropy()
    s1,s2,s3,s4,s5=_sc_markov()
    sng=_sc_ngram(); ssk=_sc_streak(); spt=_sc_point()
    sf10=_sc_freq(10); sf20=_sc_freq(20)
    smom=_sc_momentum(); srep=_sc_repeat()
    ef=max(0.3,1-e*0.4)

    w1=_aw("m1",0.06); w2=_aw("m2",0.09); w3=_aw("m3",0.12)
    w4=_aw("m4",0.13); w5=_aw("m5",0.13); wng=_aw("ng",0.18*ef)
    wsk=_aw("sk",0.10); wpt=_aw("pt",0.06); wf10=_aw("fr10",0.04)
    wf20=_aw("fr20",0.04); wmom=_aw("mom",0.03); wrep=_aw("rep",0.02)
    tw=w1+w2+w3+w4+w5+wng+wsk+wpt+wf10+wf20+wmom+wrep

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(w1*s1.get(r,0)+w2*s2.get(r,0)+w3*s3.get(r,0)+w4*s4.get(r,0)+
                w5*s5.get(r,0)+wng*sng.get(r,0)+wsk*ssk.get(r,0)+wpt*spt.get(r,0)+
                wf10*sf10.get(r,0)+wf20*sf20.get(r,0)+wmom*smom.get(r,0)+wrep*srep.get(r,0))/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred="Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    conf=max(raw["Tài"],raw["Xỉu"])
    counted=[_acc[k]["ok"]/_acc[k]["n"] for k in _acc if _acc[k]["n"]>=15]
    hist_acc=sum(counted)/len(counted) if counted else 0.5
    all_p=[_win(s1),_win(s2),_win(s3),_win(s4),_win(s5),
           _win(sng),_win(ssk),_win(spt),_win(sf10),_win(sf20),_win(smom),_win(srep)]
    dong_thuan=all_p.count(pred)/len(all_p)
    tin_cay=hist_acc*0.45+(conf-0.5)*2*0.35+dong_thuan*0.20
    tin_cay_pct=round(max(50.0,min(95.0,50+tin_cay*45)),1)

    global _prev_model
    _prev_model={"m1":_win(s1),"m2":_win(s2),"m3":_win(s3),"m4":_win(s4),"m5":_win(s5),
                 "ng":_win(sng),"sk":_win(ssk),"pt":_win(spt),"fr10":_win(sf10),
                 "fr20":_win(sf20),"mom":_win(smom),"rep":_win(srep)}
    return pred, tin_cay_pct


# ===============================
# BOT LẤY DỮ LIỆU
# ===============================
def fetch_data_loop():
    global last_processed_session_id, latest_data, _prev_pred

    while True:
        try:
            res  = requests.get(API_URL, timeout=10)
            data = res.json()
            list_data = data.get("list", [])
            if not list_data: time.sleep(2); continue

            phien    = list_data[0]
            phien_id = phien.get("id")
            if phien_id == last_processed_session_id: time.sleep(2); continue

            dices    = phien.get("dices")
            tong     = phien.get("point")
            d1,d2,d3 = dices
            ket_qua  = "Tài" if tong>=11 else "Xỉu"

            _update_stats(ket_qua)
            if len(history)>=MIN_PHIEN: _update_acc(ket_qua)

            history.append(ket_qua)
            hist_pt.append(tong)
            last_processed_session_id = phien_id

            pred, tin_cay = get_prediction()
            _prev_pred = pred

            # ── JSON GIỮ NGUYÊN FORMAT GỐC ──
            latest_data.update({
                "Phiên":          phien_id,
                "Xúc xắc 1":      d1,
                "Xúc xắc 2":      d2,
                "Xúc xắc 3":      d3,
                "Tổng":           tong,
                "Kết":            ket_qua,
                "Phiên hiện tại": phien_id + 1,
                "Dự đoán":        pred,
                "Độ tin cậy":     tin_cay,
                "ID":             "tuananh"
            })

            # ── IN TERMINAL ──
            so=len(history); cur_val,cur_len=_cur_streak()
            td=stats["tong"]
            acc_s=f"{stats['dung']}/{td} ({stats['dung']/td*100:.1f}%)" if td else "Chưa có"

            print("\n"+"="*44)
            print(f"  Phiên         : {phien_id}")
            print(f"  Xúc xắc      : {d1}  {d2}  {d3}")
            print(f"  Tổng          : {tong}")
            print(f"  Kết quả       : {ket_qua}")
            print(f"  Chuỗi         : {cur_val} x{cur_len}" if cur_val else "  Chuỗi         : --")
            print(f"  Bộ nhớ        : {so}/{MAX_PHIEN} phiên")
            print("-"*44)
            if pred=="Đang chờ":
                print(f"  Dự đoán       : Chờ thêm {MIN_PHIEN-so} phiên...")
            else:
                print(f"  Dự đoán P.{phien_id+1}  : >>> {pred} <<<")
                print(f"  Độ tin cậy    : {tin_cay}%")
                print("-"*44)
                print(f"  Đúng/Sai      : {stats['dung']}/{stats['sai']}  |  {acc_s}")
                print(f"  Chuỗi đúng   : {stats['cd']} (max {stats['max_cd']})")
                print(f"  Chuỗi sai    : {stats['cs']} (max {stats['max_cs']})")
                print("-"*44)
                print("  Accuracy 12 mô hình:")
                for lbl,key in [("Markov 1","m1"),("Markov 2","m2"),("Markov 3","m3"),
                                 ("Markov 4","m4"),("Markov 5","m5"),("N-Gram  ","ng"),
                                 ("Streak  ","sk"),("Point   ","pt"),("Freq-10 ","fr10"),
                                 ("Freq-20 ","fr20"),("Momentum","mom"),("Repeat  ","rep")]:
                    print(f"    {lbl}: {_acc_str(key)}")
            print(f"  ID            : tuananh")
            print("="*44)

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi:", e)

        time.sleep(2)


# ===============================
# CHẠY THREAD
# ===============================
threading.Thread(target=fetch_data_loop, daemon=True).start()


# ===============================
# API (GIỮ NGUYÊN)
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify({"data": latest_data})

@app.route("/", methods=["GET"])
def home():
    return "Bot Tài Xỉu – 12 AI – đang chạy"


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
