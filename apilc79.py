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
API_URL   = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MAX_PHIEN = 100
MIN_PHIEN = 50

app = Flask(__name__)

# ===============================
# DATA
# ===============================
history  = deque(maxlen=MAX_PHIEN)
hist_pt  = deque(maxlen=MAX_PHIEN)
lich_su  = []
stats    = {"tong":0,"dung":0,"sai":0,"cd":0,"cs":0,"max_cd":0,"max_cs":0}
last_phien  = None
_prev_pred  = None

latest_data = {
    "Phiên":          None,
    "Xúc xắc 1":      None,
    "Xúc xắc 2":      None,
    "Xúc xắc 3":      None,
    "Tổng":           None,
    "Kết":            None,
    "Phiên hiện tại": None,
    "Dự đoán":        "",
    "Độ tin cậy":     0,
    "Pattern":        "",
    "ID":             "tuananh"
}

# ═══════════════════════════════════════════
#  AI ENGINE – 10 MÔ HÌNH
# ═══════════════════════════════════════════

_t1 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t2 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t3 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_ng = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}
_acc = {k: {"ok":0,"n":0} for k in
        ("mkv1","mkv2","mkv3","ngram","bayes","streak","rev","cycle","mom","ma")}
_prev_model = {}


# ── 1-3: Markov Chain Model bậc 1→3 ─────────────────
def _train_markov():
    for tb in (_t1,_t2,_t3):
        for d in tb.values(): d.update({"Tài":0,"Xỉu":0})
    h=list(history)
    for i in range(len(h)-1): _t1[h[i]][h[i+1]] += 1
    for i in range(len(h)-2): _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(len(h)-3): _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1

def _s(table, key):
    d=table.get(key,{"Tài":0,"Xỉu":0}); t=d["Tài"]+d["Xỉu"]
    if not t: return {"Tài":0.0,"Xỉu":0.0}
    return {"Tài":d["Tài"]/t,"Xỉu":d["Xỉu"]/t}

def _sc_markov():
    h=list(history)
    s1=_s(_t1,h[-1])            if len(h)>=1 else {"Tài":0.0,"Xỉu":0.0}
    s2=_s(_t2,h[-2]+"|"+h[-1])  if len(h)>=2 else {"Tài":0.0,"Xỉu":0.0}
    s3=_s(_t3,"|".join(h[-3:])) if len(h)>=3 else {"Tài":0.0,"Xỉu":0.0}
    return s1,s2,s3


# ── 4: N-Gram Sequence Model ─────────────────────────
def _train_ngram():
    _ng.clear(); h=list(history)
    for ln in range(1,11):
        for i in range(len(h)-ln):
            _ng["|".join(h[i:i+ln])][h[i+ln]] += 1

def _sc_ngram():
    sc={"Tài":0.0,"Xỉu":0.0}; h=list(history)
    for ln in range(min(10,len(h)),0,-1):
        pat="|".join(h[-ln:]); d=_ng.get(pat)
        if not d: continue
        t=d["Tài"]+d["Xỉu"]
        if not t: continue
        w=ln**3; sc["Tài"]+=w*d["Tài"]/t; sc["Xỉu"]+=w*d["Xỉu"]/t
    return sc


# ── 5: Bayesian Inference Model ──────────────────────
def _sc_bayesian():
    h=list(history)
    if len(h)<8: return {"Tài":0.5,"Xỉu":0.5}
    log_odds=0.0
    # Prior: tần suất 5 phiên gần nhất
    p5=h[-5:].count("Tài")/5 if len(h)>=5 else 0.5
    if p5>0.8:   log_odds -= 1.5
    elif p5<0.2: log_odds += 1.5
    elif p5>0.6: log_odds -= 0.6
    elif p5<0.4: log_odds += 0.6
    # Evidence: streak hiện tại
    cur,ln=_cur_streak()
    if cur and ln>=3:
        log_odds += -0.6*ln if cur=="Tài" else 0.6*ln
    # Evidence: xen kẽ
    if len(h)>=4:
        alt=sum(1 for i in range(3) if h[-1-i]!=h[-2-i])
        if alt==3: log_odds += 0.4 if h[-1]=="Xỉu" else -0.4
    prob=1/(1+math.exp(-log_odds))
    return {"Tài":prob,"Xỉu":1-prob}


# ── 6: Streak Analysis Model ─────────────────────────
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
    total=ended+longer
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    other="Xỉu" if cur=="Tài" else "Tài"
    return {cur:longer/total, other:ended/total}


# ── 7: Reversal Detection Model ──────────────────────
def _sc_reversal():
    h=list(history)
    if len(h)<6: return {"Tài":0.5,"Xỉu":0.5}
    recent=h[-8:]
    sw=sum(1 for i in range(len(recent)-1) if recent[i]!=recent[i+1])
    # Xen kẽ nhiều → dự đoán đảo
    if sw>=6:
        other="Xỉu" if h[-1]=="Tài" else "Tài"
        return {other:0.72, h[-1]:0.28}
    # Chuỗi dài → dự đoán tiếp tục
    if sw<=1:
        return {h[-1]:0.68, ("Xỉu" if h[-1]=="Tài" else "Tài"):0.32}
    return {"Tài":0.5,"Xỉu":0.5}


# ── 8: Cycle Prediction Model ────────────────────────
def _sc_cycle():
    h=list(history)
    if len(h)<12: return {"Tài":0.5,"Xỉu":0.5}
    best_score=0; best_pred=None
    for cycle in range(2,8):
        if len(h)<cycle*2+1: continue
        match=sum(1 for i in range(len(h)-cycle) if h[i]==h[i+cycle])
        total=len(h)-cycle
        if total==0: continue
        score=match/total
        if score>best_score and score>0.68:
            best_score=score
            idx=len(h)%cycle
            if idx==0: idx=cycle
            best_pred=h[-idx] if idx<=len(h) else None
    if best_pred:
        return {best_pred:best_score, ("Xỉu" if best_pred=="Tài" else "Tài"):1-best_score}
    return {"Tài":0.5,"Xỉu":0.5}


# ── 9: Momentum Model ────────────────────────────────
def _sc_momentum():
    h=list(history)
    if len(h)<20: return {"Tài":0.5,"Xỉu":0.5}
    p5 =h[-5:].count("Tài")/5
    p10=h[-10:].count("Tài")/10
    p20=h[-20:].count("Tài")/20
    # Momentum ngắn vs trung vs dài hạn
    mom_short =(p5-p10)*0.6
    mom_medium=(p10-p20)*0.4
    mom=mom_short+mom_medium
    return {"Tài":max(0.0,min(1.0,0.5+mom)),"Xỉu":max(0.0,min(1.0,0.5-mom))}


# ── 10: Moving Average + Trend Strength Model ────────
def _sc_moving_avg():
    h=list(history)
    if len(h)<10: return {"Tài":0.5,"Xỉu":0.5}
    # MA ngắn (5) vs MA dài (15)
    ma5 =sum(1 for x in h[-5:]  if x=="Tài")/5
    ma15=sum(1 for x in h[-15:] if x=="Tài")/min(15,len(h)) if len(h)>=15 else 0.5
    # Trend strength = độ lệch của MA
    diff=ma5-ma15
    # MA ngắn > MA dài → xu hướng Tài đang tăng
    p_t=max(0.0,min(1.0,0.5+diff*1.5))
    return {"Tài":p_t,"Xỉu":1-p_t}


# ── Entropy ──────────────────────────────────────────
def _entropy(w=25):
    h=list(history)[-w:]; n=len(h)
    if n==0: return 1.0
    ct=h.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))


# ── Adaptive Weight ──────────────────────────────────
def _aw(key, base):
    a=_acc[key]
    if a["n"]<15: return base
    return max(0.005, base*(1+4.0*(a["ok"]/a["n"]-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_acc(actual):
    for k,p in _prev_model.items():
        if p: _acc[k]["n"]+=1; _acc[k]["ok"]+=(p==actual)

def _update_stats(actual, phien_id=None):
    global _prev_pred
    if not _prev_pred or _prev_pred=="": return
    dung=(_prev_pred==actual)
    stats["tong"]+=1
    if dung:
        stats["dung"]+=1; stats["cd"]+=1; stats["cs"]=0
        if stats["cd"]>stats["max_cd"]: stats["max_cd"]=stats["cd"]
    else:
        stats["sai"]+=1; stats["cs"]+=1; stats["cd"]=0
        if stats["cs"]>stats["max_cs"]: stats["max_cs"]=stats["cs"]

def _acc_str(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"


# ===============================
# DỰ ĐOÁN – 10 MÔ HÌNH
# ===============================
def du_doan_ai():
    if len(history)<MIN_PHIEN:
        return "", 0

    _train_markov(); _train_ngram(); _train_streak()
    e=_entropy()
    s1,s2,s3=_sc_markov()
    sng=_sc_ngram(); sbay=_sc_bayesian()
    ssk=_sc_streak(); srev=_sc_reversal()
    scyc=_sc_cycle(); smom=_sc_momentum(); sma=_sc_moving_avg()
    ef=max(0.3,1-e*0.4)

    w1=_aw("mkv1",0.09); w2=_aw("mkv2",0.12); w3=_aw("mkv3",0.13)
    wng=_aw("ngram",0.18*ef); wbay=_aw("bayes",0.12)
    wsk=_aw("streak",0.10); wrev=_aw("rev",0.08)
    wcyc=_aw("cycle",0.08); wmom=_aw("mom",0.06); wma=_aw("ma",0.04)
    tw=w1+w2+w3+wng+wbay+wsk+wrev+wcyc+wmom+wma

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(w1*s1.get(r,0)+w2*s2.get(r,0)+w3*s3.get(r,0)+
                wng*sng.get(r,0)+wbay*sbay.get(r,0)+
                wsk*ssk.get(r,0)+wrev*srev.get(r,0)+
                wcyc*scyc.get(r,0)+wmom*smom.get(r,0)+wma*sma.get(r,0))/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred="Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    conf=max(raw["Tài"],raw["Xỉu"])

    # Accuracy lịch sử thực
    counted=[_acc[k]["ok"]/_acc[k]["n"] for k in _acc if _acc[k]["n"]>=10]
    hist_acc=sum(counted)/len(counted) if counted else 0.5

    # Đồng thuận
    all_p=[_win(s1),_win(s2),_win(s3),_win(sng),_win(sbay),
           _win(ssk),_win(srev),_win(scyc),_win(smom),_win(sma)]
    dong_thuan=all_p.count(pred)/len(all_p)

    # Độ tin cậy 50-100% thật
    raw_conf=(conf-0.5)*2
    acc_bonus=max(0,hist_acc-0.5)*2
    thuan_bonus=max(0,dong_thuan-0.5)*2
    score=raw_conf*0.50+acc_bonus*0.30+thuan_bonus*0.20
    tin_cay=round(max(50.0,min(100.0,50+score*50)),1)

    global _prev_model
    _prev_model={
        "mkv1":_win(s1),"mkv2":_win(s2),"mkv3":_win(s3),
        "ngram":_win(sng),"bayes":_win(sbay),"streak":_win(ssk),
        "rev":_win(srev),"cycle":_win(scyc),"mom":_win(smom),"ma":_win(sma)
    }
    return pred, tin_cay


# ===============================
# FETCH DATA
# ===============================
def fetch_loop():
    global last_phien, latest_data, _prev_pred

    while True:
        try:
            res  = requests.get(API_URL, timeout=5)
            data = res.json()
            ds   = data.get("list", [])
            if not ds: time.sleep(2); continue

            item  = ds[0]
            phien = item.get("id")
            if phien==last_phien: time.sleep(2); continue

            d1,d2,d3 = item.get("dices")
            tong      = item.get("point")
            ket       = "Tài" if tong>=11 else "Xỉu"

            _update_stats(ket, phien)
            if len(history)>=MIN_PHIEN: _update_acc(ket)

            history.append(ket)
            hist_pt.append(tong)

            pattern = "".join(
                ["T" if x=="Tài" else "X" for x in list(history)[-20:]]
            )

            du_doan, do_tin_cay = du_doan_ai()
            _prev_pred = du_doan

            so=len(history); cur_val,cur_len=_cur_streak()
            td=stats["tong"]
            acc_s=f"{stats['dung']}/{td} ({stats['dung']/td*100:.1f}%)" if td else "Chưa có"

            latest_data = {
                "Phiên":          phien,
                "Xúc xắc 1":      d1,
                "Xúc xắc 2":      d2,
                "Xúc xắc 3":      d3,
                "Tổng":           tong,
                "Kết":            ket,
                "Phiên hiện tại": phien+1,
                "Dự đoán":        du_doan,
                "Độ tin cậy":     do_tin_cay,
                "Pattern":        pattern,
                "ID":             "tuananh"
            }

            print(f"Phiên   : {phien}")
            print(f"Xúc xắc : {d1}  {d2}  {d3}")
            print(f"Tổng    : {tong}  |  Kết: {ket}")
            print(f"Pattern : {pattern}")
            if not du_doan:
                print(f"Dự đoán : Chờ {MIN_PHIEN-so} phiên...")
            else:
                print(f"Dự đoán : >>> {du_doan} <<<  ({do_tin_cay}%)")
                print(f"Đúng/Sai: {stats['dung']}/{stats['sai']}  |  {acc_s}")
            print("-"*35)

            last_phien = phien

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi:", e)

        time.sleep(2)


# ===============================
# CHẠY THREAD
# ===============================
threading.Thread(target=fetch_loop, daemon=True).start()


# ===============================
# API
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify({"data": latest_data})

@app.route("/api/lichsu", methods=["GET"])
def api_lichsu():
    td=stats["tong"]
    return jsonify({
        "tong":    stats["tong"],
        "dung":    stats["dung"],
        "sai":     stats["sai"],
        "ty_le":   f"{stats['dung']/td*100:.1f}%" if td else "0%",
        "max_cd":  stats["max_cd"],
        "max_cs":  stats["max_cs"],
        "lich_su": lich_su[-20:]
    })


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
