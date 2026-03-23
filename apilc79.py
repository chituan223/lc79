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
API_URL   = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MIN_PHIEN = 20
MAX_PHIEN = 200

app = Flask(__name__)

# ===============================
# LỊCH SỬ
# ===============================
history   = deque(maxlen=MAX_PHIEN)   # "Tài"/"Xỉu"
hist_pt   = deque(maxlen=MAX_PHIEN)   # tổng điểm
hist_dice = deque(maxlen=MAX_PHIEN)   # (d1,d2,d3)

# ===============================
# THỐNG KÊ DỰ ĐOÁN
# ===============================
stats = {"tong":0,"dung":0,"sai":0,"cd":0,"cs":0,"max_cd":0,"max_cs":0}
_prev_pred = None

# ══════════════════════════════════════════════════════
#  20 MÔ HÌNH AI – CẤU TRÚC DỮ LIỆU
# ══════════════════════════════════════════════════════

# Markov bậc 1–6
_mk = [{} for _ in range(7)]   # _mk[1..6]

# N-gram table
_ng = {}

# Streak distribution
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}

# Transition matrix 2x2 (Tài→Tài, Tài→Xỉu, Xỉu→Tài, Xỉu→Xỉu)
_trans = {"TT":0,"TX":0,"XT":0,"XX":0}

# Bayesian posterior (Beta distribution: alpha, beta)
_bayes = {"Tài": 1.0, "Xỉu": 1.0}   # uniform prior

# Autocorrelation lags 1–15
_acf = {}

# Pattern cluster: last 4 bits → next
_pc = {}

# Model accuracy tracking (20 models)
_MODEL_KEYS = [
    "m1","m2","m3","m4","m5","m6",
    "ng","sk","pt",
    "fr10","fr20","fr50",
    "mom","rep","zig",
    "bay","acf","ldecay","pclust","meta"
]
_acc = {k: {"ok":0,"n":0,"streak_ok":0,"streak_fail":0} for k in _MODEL_KEYS}
_prev_model = {}

# Recent window for meta-learner (last 30 predictions per model)
_model_recent = {k: deque(maxlen=30) for k in _MODEL_KEYS}


# ══════════════════════════════════════════════════════
#  HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════

def _win(sc):
    return "Tài" if sc.get("Tài",0) >= sc.get("Xỉu",0) else "Xỉu"

def _norm(sc):
    s = sc.get("Tài",0) + sc.get("Xỉu",0)
    if s <= 0: return {"Tài":0.5,"Xỉu":0.5}
    return {"Tài":sc["Tài"]/s, "Xỉu":sc["Xỉu"]/s}

def _half(v=0.5):
    return {"Tài":v,"Xỉu":1-v}

def _acc_str(key):
    a = _acc[key]
    if a["n"] == 0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.1f}%)"

def _recent_acc(key, window=20):
    """Accuracy trên window phiên gần nhất."""
    buf = list(_model_recent[key])[-window:]
    if not buf: return 0.5
    return sum(buf) / len(buf)

def _aw(key, base, boost=5.0):
    """Adaptive weight dựa trên recent accuracy."""
    a = _acc[key]
    if a["n"] < 15: return base
    rate = _recent_acc(key, 30)
    return max(0.001, base * (1 + boost * (rate - 0.5)))

def _entropy(w=50):
    h = list(history)[-w:]
    n = len(h)
    if n == 0: return 1.0
    ct = h.count("Tài"); cx = n - ct
    if ct == 0 or cx == 0: return 0.0
    pt, px = ct/n, cx/n
    return -(pt*math.log2(pt) + px*math.log2(px))

def _cur_streak():
    h = list(history)
    if not h: return None, 0
    cur = h[-1]; cnt = 1
    for r in reversed(h[:-1]):
        if r == cur: cnt += 1
        else: break
    return cur, cnt


# ══════════════════════════════════════════════════════
#  TRAINING – cập nhật toàn bộ sau mỗi phiên
# ══════════════════════════════════════════════════════

def _train_all():
    h = list(history)
    n = len(h)
    if n < 2: return

    # ── MARKOV bậc 1–6 ──────────────────────────────
    for order in range(1, 7):
        table = {}
        for i in range(n - order):
            key = "|".join(h[i:i+order])
            nxt = h[i+order]
            if key not in table: table[key] = {"Tài":0,"Xỉu":0}
            table[key][nxt] += 1
        _mk[order] = table

    # ── N-GRAM bậc 1–15 ─────────────────────────────
    _ng.clear()
    for ln in range(1, 16):
        for i in range(n - ln):
            pat = "|".join(h[i:i+ln])
            if pat not in _ng: _ng[pat] = {"Tài":0,"Xỉu":0}
            _ng[pat][h[i+ln]] += 1

    # ── STREAK DISTRIBUTION ──────────────────────────
    for d in _sd.values(): d.clear()
    cur, cnt = h[0], 1
    for r in h[1:]:
        if r == cur: cnt += 1
        else: _sd[cur][cnt] += 1; cur, cnt = r, 1
    _sd[cur][cnt] += 1

    # ── TRANSITION MATRIX ────────────────────────────
    _trans.update({"TT":0,"TX":0,"XT":0,"XX":0})
    for i in range(n-1):
        key = ("T" if h[i]=="Tài" else "X") + ("T" if h[i+1]=="Tài" else "X")
        _trans[key] += 1

    # ── BAYESIAN POSTERIOR (sliding window 80 phiên) ─
    recent80 = h[-80:]
    ct = recent80.count("Tài"); cx = len(recent80) - ct
    _bayes["Tài"] = 1.0 + ct    # Beta(alpha, beta) – Jeffrey's prior + data
    _bayes["Xỉu"] = 1.0 + cx

    # ── AUTOCORRELATION lags 1–15 ────────────────────
    if n >= 20:
        enc = [1 if x=="Tài" else -1 for x in h]
        mean_e = sum(enc) / len(enc)
        var    = sum((x - mean_e)**2 for x in enc)
        if var > 0:
            for lag in range(1, 16):
                cov = sum((enc[i]-mean_e)*(enc[i+lag]-mean_e) for i in range(n-lag))
                _acf[lag] = cov / var
        else:
            for lag in range(1, 16): _acf[lag] = 0.0

    # ── PATTERN CLUSTER (4-bit window → next) ────────
    _pc.clear()
    for i in range(n - 4):
        pat = tuple(h[i:i+4])
        nxt = h[i+4]
        if pat not in _pc: _pc[pat] = {"Tài":0,"Xỉu":0}
        _pc[pat][nxt] += 1


# ══════════════════════════════════════════════════════
#  20 MÔ HÌNH SCORING
# ══════════════════════════════════════════════════════

# MÔ HÌNH 1–6: MARKOV CHAIN bậc 1→6
def _sc_markov(order):
    h = list(history)
    if len(h) < order: return _half()
    key = "|".join(h[-order:])
    d   = _mk[order].get(key)
    if not d: return _half()
    return _norm(d)


# MÔ HÌNH 7: N-GRAM tối đa 15, trọng số bậc 5
def _sc_ngram():
    sc = {"Tài":0.0,"Xỉu":0.0}
    h  = list(history)
    total_w = 0.0
    for ln in range(min(15, len(h)), 0, -1):
        pat = "|".join(h[-ln:])
        d   = _ng.get(pat)
        if not d: continue
        t = d["Tài"] + d["Xỉu"]
        if not t: continue
        w = ln ** 5   # bậc 5 – ưu tiên mạnh pattern dài
        sc["Tài"]  += w * d["Tài"] / t
        sc["Xỉu"]  += w * d["Xỉu"] / t
        total_w    += w
    if not total_w: return _half()
    return _norm(sc)


# MÔ HÌNH 8: STREAK REVERSAL – có trọng số độ dài
def _sc_streak():
    cur, ln = _cur_streak()
    if not cur: return _half()
    dist   = _sd[cur]
    ended  = sum(v * (k ** 1.8) for k, v in dist.items() if k <= ln)
    longer = sum(v * (k ** 1.8) for k, v in dist.items() if k > ln)
    total  = ended + longer
    if not total: return _half()
    other  = "Xỉu" if cur == "Tài" else "Tài"
    return {cur: longer/total, other: ended/total}


# MÔ HÌNH 9: POINT BIAS – phân tích tổng điểm thực + slope
def _sc_point(w=40):
    pts = list(hist_pt)
    if len(pts) < 10: return _half()
    recent = pts[-w:]
    avg    = sum(recent) / len(recent)
    n      = len(recent)
    if n >= 8:
        half1  = sum(recent[:n//2]) / (n//2)
        half2  = sum(recent[n//2:]) / (n - n//2)
        slope  = (half2 - half1) / 10.5
    else:
        slope = 0
    # Tổng 3–18, ngưỡng Tài=11. Điểm TB 10.5 → cân bằng.
    p_t = max(0.0, min(1.0, (avg - 10.5) / 7.5 + slope * 0.12))
    return {"Tài": 0.5 + p_t * 0.5, "Xỉu": 0.5 - p_t * 0.5}


# MÔ HÌNH 10: FREQUENCY WINDOW 10 phiên
def _sc_freq10():
    h = list(history)
    if len(h) < 10: return _half()
    ct = h[-10:].count("Tài") / 10
    # Mean-reversion: nếu Tài nhiều → dự Xỉu và ngược lại
    return {"Tài": 1-ct, "Xỉu": ct}


# MÔ HÌNH 11: FREQUENCY WINDOW 20 phiên
def _sc_freq20():
    h = list(history)
    if len(h) < 20: return _half()
    ct = h[-20:].count("Tài") / 20
    return {"Tài": 1-ct, "Xỉu": ct}


# MÔ HÌNH 12: FREQUENCY WINDOW 50 phiên
def _sc_freq50():
    h = list(history)
    w = min(50, len(h))
    if w < 15: return _half()
    ct = h[-w:].count("Tài") / w
    return {"Tài": 1-ct, "Xỉu": ct}


# MÔ HÌNH 13: MOMENTUM đa tầng (5/15/30/50)
def _sc_momentum():
    h = list(history)
    n = len(h)
    if n < 30: return _half()
    p5  = h[-5:].count("Tài")  / 5
    p15 = h[-15:].count("Tài") / 15
    p30 = h[-30:].count("Tài") / 30
    p50 = (h[-50:].count("Tài") / 50) if n >= 50 else p30
    mom = (p5-p15)*0.50 + (p15-p30)*0.30 + (p30-p50)*0.20
    p_t = max(0.0, min(1.0, 0.5 + mom * 1.5))
    return {"Tài": p_t, "Xỉu": 1-p_t}


# MÔ HÌNH 14: REPETITION PATTERN chu kỳ 2–7
def _sc_repeat():
    h = list(history)
    if len(h) < 10: return _half()
    sc = {"Tài":0.0,"Xỉu":0.0}
    for cycle in range(2, 8):
        if len(h) < cycle * 3: continue
        match = 0
        for i in range(1, min(5, cycle+1)):
            if h[-i] == h[-i-cycle]: match += 1
        if match >= 2:
            pred = h[-cycle]
            w    = (match ** 2) * cycle
            sc[pred] += w
    total = sc["Tài"] + sc["Xỉu"]
    if not total: return _half()
    return _norm(sc)


# MÔ HÌNH 15: ZIGZAG / ALTERNATING DETECTOR
def _sc_zigzag():
    h = list(history)
    if len(h) < 6: return _half()
    # Đếm số lần đổi chiều trong 20 phiên cuối
    w     = min(20, len(h))
    seg   = h[-w:]
    flips = sum(1 for i in range(len(seg)-1) if seg[i] != seg[i+1])
    flip_rate = flips / (w - 1)   # tỉ lệ đổi chiều [0,1]
    last  = h[-1]
    if flip_rate >= 0.6:     # chuỗi đang zíc-zắc → đổi chiều
        other = "Xỉu" if last == "Tài" else "Tài"
        return {other: 0.5 + (flip_rate - 0.6) * 2.5, last: 1 - (0.5 + (flip_rate-0.6)*2.5)}
    elif flip_rate <= 0.3:   # chuỗi đang liên tiếp → giữ
        return {last: 0.5 + (0.3 - flip_rate) * 2.5, ("Xỉu" if last=="Tài" else "Tài"): 1-(0.5+(0.3-flip_rate)*2.5)}
    return _half()


# MÔ HÌNH 16: BAYESIAN POSTERIOR (Beta distribution)
def _sc_bayesian():
    a = _bayes["Tài"]
    b = _bayes["Xỉu"]
    mean_t = a / (a + b)
    return {"Tài": mean_t, "Xỉu": 1 - mean_t}


# MÔ HÌNH 17: AUTOCORRELATION – phát hiện chu kỳ ẩn
def _sc_acf():
    h = list(history)
    if len(h) < 20 or not _acf: return _half()
    last = h[-1]
    score = {"Tài":0.0,"Xỉu":0.0}
    for lag, corr in _acf.items():
        if len(h) <= lag: continue
        ref = h[-(lag+1)]   # kết quả cách đây `lag` phiên
        if abs(corr) < 0.05: continue
        if corr > 0:   # xu hướng lặp lại theo lag
            pred = ref
        else:          # xu hướng đảo ngược theo lag
            pred = "Xỉu" if ref == "Tài" else "Tài"
        score[pred] += abs(corr)
    total = score["Tài"] + score["Xỉu"]
    if not total: return _half()
    return _norm(score)


# MÔ HÌNH 18: LOCAL DECAY FREQUENCY
# Mỗi kết quả lịch sử có trọng số giảm dần theo thời gian (exp decay)
def _sc_local_decay(lam=0.04):
    h = list(history)
    n = len(h)
    if n < 5: return _half()
    wt = wxt = 0.0
    for i, r in enumerate(reversed(h)):
        w = math.exp(-lam * i)
        if r == "Tài": wt += w
        else:          wxt += w
    total = wt + wxt
    if not total: return _half()
    return {"Tài": wt/total, "Xỉu": wxt/total}


# MÔ HÌNH 19: PATTERN CLUSTER (4-bit → next)
def _sc_pcluster():
    h = list(history)
    if len(h) < 5: return _half()
    pat = tuple(h[-4:])
    d   = _pc.get(pat)
    if not d: return _half()
    return _norm(d)


# MÔ HÌNH 20: META-LEARNER
# Ensemble of the 19 models, weighted by recent 20-phiên accuracy
def _sc_meta(scores_dict):
    """scores_dict: {model_key: score_dict} cho 19 models đã tính."""
    sc = {"Tài":0.0,"Xỉu":0.0}
    tw = 0.0
    for k, scd in scores_dict.items():
        ra = _recent_acc(k, 20)
        # Chỉ tin model nếu recent accuracy > 50%
        w  = max(0.0, ra - 0.45) * 10
        if w <= 0: continue
        sc["Tài"]  += w * scd.get("Tài",0)
        sc["Xỉu"]  += w * scd.get("Xỉu",0)
        tw += w
    if tw < 0.1: return _half()
    return _norm(sc)


# ══════════════════════════════════════════════════════
#  CẬP NHẬT ACCURACY
# ══════════════════════════════════════════════════════

def _update_model_acc(actual):
    for k, p in _prev_model.items():
        if p:
            correct = (p == actual)
            _acc[k]["n"]   += 1
            _acc[k]["ok"]  += int(correct)
            _model_recent[k].append(int(correct))


# ══════════════════════════════════════════════════════
#  CẬP NHẬT THỐNG KÊ TỔNG
# ══════════════════════════════════════════════════════

def _update_stats(actual):
    global _prev_pred
    if not _prev_pred or _prev_pred == "Đang chờ": return
    stats["tong"] += 1
    if _prev_pred == actual:
        stats["dung"] += 1; stats["cd"] += 1; stats["cs"] = 0
        if stats["cd"] > stats["max_cd"]: stats["max_cd"] = stats["cd"]
    else:
        stats["sai"] += 1; stats["cs"] += 1; stats["cd"] = 0
        if stats["cs"] > stats["max_cs"]: stats["max_cs"] = stats["cs"]


# ══════════════════════════════════════════════════════
#  DỰ ĐOÁN CHÍNH
# ══════════════════════════════════════════════════════

def get_prediction():
    if len(history) < MIN_PHIEN:
        return "Đang chờ", 0.0, {}

    _train_all()

    e  = _entropy()
    ef = max(0.3, 1.0 - e * 0.45)    # entropy factor giảm trọng số khi hỗn loạn

    # ── Tính điểm 20 mô hình ────────────────────────
    sc = {
        "m1":    _sc_markov(1),
        "m2":    _sc_markov(2),
        "m3":    _sc_markov(3),
        "m4":    _sc_markov(4),
        "m5":    _sc_markov(5),
        "m6":    _sc_markov(6),
        "ng":    _sc_ngram(),
        "sk":    _sc_streak(),
        "pt":    _sc_point(),
        "fr10":  _sc_freq10(),
        "fr20":  _sc_freq20(),
        "fr50":  _sc_freq50(),
        "mom":   _sc_momentum(),
        "rep":   _sc_repeat(),
        "zig":   _sc_zigzag(),
        "bay":   _sc_bayesian(),
        "acf":   _sc_acf(),
        "ldecay":_sc_local_decay(),
        "pclust":_sc_pcluster(),
    }

    # Model 20: Meta-learner dùng điểm của 19 model trên
    sc["meta"] = _sc_meta(sc)

    # ── Base weights ────────────────────────────────
    base_w = {
        "m1":    0.030,
        "m2":    0.050,
        "m3":    0.070,
        "m4":    0.080,
        "m5":    0.080,
        "m6":    0.070,
        "ng":    0.130 * ef,
        "sk":    0.075,
        "pt":    0.040,
        "fr10":  0.025,
        "fr20":  0.025,
        "fr50":  0.020,
        "mom":   0.030,
        "rep":   0.025,
        "zig":   0.030,
        "bay":   0.035,
        "acf":   0.040,
        "ldecay":0.045,
        "pclust":0.060,
        "meta":  0.080,
    }

    # ── Adaptive weights ─────────────────────────────
    aw = {k: _aw(k, base_w[k]) for k in _MODEL_KEYS}
    tw = sum(aw.values())
    if tw <= 0: tw = 1.0

    # ── Weighted ensemble ────────────────────────────
    raw = {"Tài":0.0,"Xỉu":0.0}
    for k in _MODEL_KEYS:
        for r in ("Tài","Xỉu"):
            raw[r] += aw[k] * sc[k].get(r, 0)

    raw = {r: v/tw for r, v in raw.items()}
    s   = raw["Tài"] + raw["Xỉu"]
    if s > 0: raw = {r: v/s for r, v in raw.items()}
    else:     raw = {"Tài":0.5,"Xỉu":0.5}

    pred = "Tài" if raw["Tài"] >= raw["Xỉu"] else "Xỉu"
    conf = max(raw["Tài"], raw["Xỉu"])

    # ── Độ tin cậy thực ─────────────────────────────
    # 1. Lịch sử accuracy các model
    counted = [_acc[k]["ok"] / _acc[k]["n"]
               for k in _MODEL_KEYS if _acc[k]["n"] >= 15]
    hist_acc = sum(counted)/len(counted) if counted else 0.5

    # 2. Đồng thuận 20 model
    all_preds     = [_win(sc[k]) for k in _MODEL_KEYS]
    dong_thuan    = all_preds.count(pred) / len(all_preds)

    # 3. Entropy factor
    e_score = 1.0 - e   # cao → dữ liệu có pattern rõ

    # 4. Meta accuracy gần đây
    meta_acc = _recent_acc("meta", 20)

    tin_cay = (
        hist_acc  * 0.35 +
        (conf-0.5)*2 * 0.30 +
        dong_thuan * 0.20 +
        e_score    * 0.10 +
        meta_acc   * 0.05
    )
    tin_cay_pct = round(max(50.0, min(96.0, 50 + tin_cay * 46)), 1)

    # ── Lưu dự đoán từng model ───────────────────────
    global _prev_model
    _prev_model = {k: _win(sc[k]) for k in _MODEL_KEYS}

    return pred, tin_cay_pct, sc


# ══════════════════════════════════════════════════════
#  BOT NỀN
# ══════════════════════════════════════════════════════
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

            dices    = phien.get("dices")
            tong     = phien.get("point")
            d1, d2, d3 = dices
            ket      = "Tài" if tong >= 11 else "Xỉu"

            # Cập nhật accuracy TRƯỚC khi thêm kết quả mới
            _update_stats(ket)
            if len(history) >= MIN_PHIEN:
                _update_model_acc(ket)

            history.append(ket)
            hist_pt.append(tong)
            hist_dice.append((d1, d2, d3))

            pred, tin_cay, sc = get_prediction()
            _prev_pred = pred
            last_id    = phien_id
            phien_tiep = phien_id + 1
            so_phien   = len(history)
            cur_val, cur_len = _cur_streak()
            tong_dd  = stats["tong"]
            acc_tong = (f"{stats['dung']}/{tong_dd} "
                        f"({stats['dung']/tong_dd*100:.1f}%)"
                        if tong_dd else "Chưa có")

            # ── JSON ─────────────────────────────────
            latest_data = {
                "Phiên":          phien_id,
                "Xúc_xắc_1":      d1,
                "Xúc_xắc_2":      d2,
                "Xúc_xắc_3":      d3,
                "Tổng":           tong,
                "Kết":            ket,
                "phien_hien_tai": phien_tiep,
                "Dự_đoán":        pred,
                "Độ_tin_cậy":     f"{tin_cay}%",
                "Tỷ_lệ_đúng":     acc_tong,
                "ID":             "tuananh"
            }

            
            
            print(f"\n{SEP}")
            print(f"  Phiên          : {phien_id}")
            print(f"  Xúc xắc        : {d1}  {d2}  {d3}")
            print(f"  Tổng           : {tong}")
            print(f"  Kết quả        : {ket}")
            if cur_val:
                print(f"  Chuỗi hiện tại : {cur_val} × {cur_len}")
            print(f"  Bộ nhớ         : {so_phien}/{MAX_PHIEN} phiên")
            print(sep)

            if pred == "Đang chờ":
                print(f"  Dự đoán : Chờ thêm {MIN_PHIEN-so_phien} phiên nữa...")
            else:
                print(f"  ▶ Dự đoán P.{phien_tiep:<9}: >>> {pred} <<<")
                print(f"  ▶ Độ tin cậy    : {tin_cay}%")
                print(sep)
                print(f"  Thống kê tổng:")
                print(f"    Tổng đã đoán  : {tong_dd} phiên")
                print(f"    Đúng / Sai    : {stats['dung']} / {stats['sai']}")
                print(f"    Tỷ lệ đúng    : {acc_tong}")
                print(f"    Chuỗi đúng    : {stats['cd']} (max {stats['max_cd']})")
                print(f"    Chuỗi sai     : {stats['cs']} (max {stats['max_cs']})")
                print(sep)
                print("  Accuracy 20 mô hình AI:")
                labels = [
                    ("Markov 1  ","m1"),("Markov 2  ","m2"),
                    ("Markov 3  ","m3"),("Markov 4  ","m4"),
                    ("Markov 5  ","m5"),("Markov 6  ","m6"),
                    ("N-Gram    ","ng"),("Streak    ","sk"),
                    ("PointBias ","pt"),("Freq-10   ","fr10"),
                    ("Freq-20   ","fr20"),("Freq-50   ","fr50"),
                    ("Momentum  ","mom"),("Repeat    ","rep"),
                    ("Zigzag    ","zig"),("Bayesian  ","bay"),
                    ("AutoCorr  ","acf"),("Decay     ","ldecay"),
                    ("PatClust  ","pclust"),("Meta-AI   ","meta"),
                ]
                for lbl, key in labels:
                    pred_k = _win(sc.get(key, {}))
                    mark   = "✓" if pred_k == pred else " "
                    print(f"    [{mark}] {lbl}: {_acc_str(key)}")

            print(f"  ID             : tuananh")
            print(SEP)

        except Exception as err:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi: {err}")

        time.sleep(2)


# ══════════════════════════════════════════════════════
#  FLASK API
# ══════════════════════════════════════════════════════
threading.Thread(target=fetch_data_loop, daemon=True).start()

@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    if latest_data:
        return jsonify({"data": latest_data})
    return jsonify({"status": "Đang khởi động..."})


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
