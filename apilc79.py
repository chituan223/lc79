import requests
import time
import threading
import math
from flask import Flask, jsonify
from collections import deque, Counter
from datetime import datetime

# ===============================
# CẤU HÌNH
# ===============================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MAX_PHIEN = 60
MIN_PHIEN = 30

app = Flask(__name__)

# ===============================
# DATA
# ===============================
history_tx = deque(maxlen=MAX_PHIEN)   # Tài/Xỉu
history_pt = deque(maxlen=MAX_PHIEN)   # Điểm
history_id = deque(maxlen=MAX_PHIEN)   # ID phiên
history_dice = deque(maxlen=MAX_PHIEN) # Chi tiết xúc xắc

# Stats theo dõi độ chính xác thật
stats = {
    "tong": 0, "dung": 0, "sai": 0,
    "max_tai": 0, "max_xiu": 0,
    "streak_tai": 0, "streak_xiu": 0
}

# Lưu dự đoán để đánh giá
pred_log = deque(maxlen=MAX_PHIEN)  # {phien, prediction, actual, correct}

last_phien = None
last_prediction = None

latest_data = {
    "Phiên": None,
    "Xúc xắc 1": None, "Xúc xắc 2": None, "Xúc xắc 3": None,
    "Tổng": None, "Kết": None,
    "Phiên hiện tại": None,
    "Dự đoán": "Khởi động...",
    "Độ tin cậy": 0,
    "Pattern": "",
    "Tỷ lệ đúng": "0% (0/0)",
    "Chuỗi hiện tại": "",
    "Max chuỗi Tài": 0,
    "Max chuỗi Xỉu": 0,
    "AI Models": {},
    "Phân tích": "",
    "ID": "tuananh"
}

# ===============================
# AI ENGINE - ĐA MÔ HÌNH
# ===============================

def encode(tx_list):
    """Chuyển Tài/Xỉu thành T/X"""
    return "".join(["T" if x == "Tài" else "X" for x in tx_list])

def decode(c):
    return "Tài" if c == "T" else "Xỉu"

# ---------- 1. MARKOV CHAIN ----------
class MarkovChain:
    def predict(self, tx_list, order=3):
        if len(tx_list) < order + 5:
            return None, 0
        s = encode(tx_list)
        transitions = {}
        for i in range(len(s) - order):
            state = s[i:i+order]
            nxt = s[i+order]
            if state not in transitions:
                transitions[state] = Counter()
            transitions[state][nxt] += 1
        
        current = s[-order:]
        if current not in transitions:
            return None, 0
        
        counts = transitions[current]
        total = sum(counts.values())
        prob_t = counts.get("T", 0) / total
        prob_x = counts.get("X", 0) / total
        
        pred = "Tài" if prob_t > prob_x else "Xỉu"
        conf = max(prob_t, prob_x) * 100
        return pred, conf

# ---------- 2. N-GRAM MODEL ----------
class NGramModel:
    def predict(self, tx_list, max_n=5):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        votes = Counter()
        confidences = []
        
        for n in range(2, min(max_n + 1, len(s)//2)):
            if len(s) < n + 2:
                continue
            grams = Counter()
            for i in range(len(s) - n):
                gram = s[i:i+n]
                nxt = s[i+n]
                grams[(gram, nxt)] += 1
            
            current = s[-n:]
            t_score = sum(grams.get((current, "T"), 0) for _ in range(1))
            x_score = sum(grams.get((current, "X"), 0) for _ in range(1))
            
            # Tính điểm thực
            t_total = sum(v for (g, c), v in grams.items() if g == current and c == "T")
            x_total = sum(v for (g, c), v in grams.items() if g == current and c == "X")
            total = t_total + x_total
            
            if total > 0:
                if t_total > x_total:
                    votes["Tài"] += n  # N càng cao càng tin cậy
                    confidences.append(t_total / total * 100)
                elif x_total > t_total:
                    votes["Xỉu"] += n
                    confidences.append(x_total / total * 100)
        
        if not votes:
            return None, 0
        
        pred = votes.most_common(1)[0][0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 50
        return pred, min(avg_conf, 95)

# ---------- 3. PATTERN DETECTION ----------
class PatternModel:
    def __init__(self):
        self.patterns = {
            "bệt": self._detect_streak,
            "1-1": self._detect_alt,
            "2-2": self._detect_double,
            "3-3": self._detect_triple,
            "cân_bằng": self._detect_balance,
            "đảo_sau_bệt": self._detect_reversal_after_streak,
            "lặp_chu_kỳ": self._detect_cycle
        }
    
    def _detect_streak(self, s):
        if len(s) < 3:
            return None
        last = s[-1]
        count = 0
        for c in reversed(s):
            if c == last:
                count += 1
            else:
                break
        if count >= 3:
            opposite = "X" if last == "T" else "T"
            return decode(opposite), min(50 + count * 8, 90), f"Bệt {decode(last)} x{count}"
        return None
    
    def _detect_alt(self, s):
        if len(s) < 4:
            return None
        recent = s[-6:]
        is_alt = all(recent[i] != recent[i+1] for i in range(len(recent)-1))
        if is_alt:
            next_c = "X" if recent[-1] == "T" else "T"
            return decode(next_c), 75, "Cầu 1-1"
        return None
    
    def _detect_double(self, s):
        if len(s) < 6:
            return None
        recent = s[-8:]
        for i in range(0, len(recent)-3, 2):
            if not (recent[i]==recent[i+1] and recent[i+2]==recent[i+3] and recent[i]!=recent[i+2]):
                return None
        next_expected = recent[0] if len(recent) % 4 == 0 else recent[2]
        return decode(next_expected), 80, "Cầu 2-2"
    
    def _detect_triple(self, s):
        if len(s) < 9:
            return None
        recent = s[-12:]
        for i in range(0, len(recent)-5, 3):
            if not (recent[i]==recent[i+1]==recent[i+2] and recent[i+3]==recent[i+4]==recent[i+5] and recent[i]!=recent[i+3]):
                return None
        return decode(recent[0]), 85, "Cầu 3-3"
    
    def _detect_balance(self, s):
        if len(s) < 20:
            return None
        recent = s[-20:]
        t_count = recent.count("T")
        t_pct = t_count / 20
        if t_pct > 0.7:
            return "Xỉu", min(50 + (t_pct-0.5)*100, 90), f"Tài thiên lệch {t_pct:.0%}"
        elif t_pct < 0.3:
            return "Tài", min(50 + (0.5-t_pct)*100, 90), f"Xỉu thiên lệch {1-t_pct:.0%}"
        return None
    
    def _detect_reversal_after_streak(self, s):
        if len(s) < 5:
            return None
        # Sau bệt 3+ thường đảo
        streak = 0
        last = s[-1]
        for c in reversed(s):
            if c == last:
                streak += 1
            else:
                break
        if streak >= 3:
            opposite = "X" if last == "T" else "T"
            return decode(opposite), 70, f"Đảo sau bệt {streak}"
        return None
    
    def _detect_cycle(self, s):
        if len(s) < 8:
            return None
        # Tìm chu kỳ lặp lại
        for cycle_len in range(2, min(6, len(s)//2)):
            cycle = s[-cycle_len:]
            if s[-cycle_len*2:-cycle_len] == cycle:
                next_in_cycle = cycle[0]
                return decode(next_in_cycle), 80, f"Chu kỳ {cycle_len}"
        return None
    
    def predict(self, tx_list):
        s = encode(tx_list)
        results = []
        for name, detector in self.patterns.items():
            result = detector(s)
            if result:
                pred, conf, reason = result
                results.append((pred, conf, reason, name))
        
        if not results:
            return None, 0, "Không nhận diện pattern"
        
        # Chọn pattern có độ tin cậy cao nhất
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0], results[0][1], results[0][2]

# ---------- 4. STREAK ANALYSIS ----------
class StreakModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        
        # Phân tích độ dài chuỗi trung bình
        streaks = []
        current = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                current += 1
            else:
                streaks.append(current)
                current = 1
        streaks.append(current)
        
        avg_streak = sum(streaks) / len(streaks)
        current_streak = streaks[-1]
        
        # Nếu chuỗi hiện tại > trung bình + 1 -> đảo
        if current_streak > avg_streak + 0.5:
            opposite = "Xỉu" if s[-1] == "T" else "Tài"
            conf = min(50 + (current_streak - avg_streak) * 15, 90)
            return opposite, conf, f"Streak {current_streak} > avg {avg_streak:.1f}"
        
        # Nếu chuỗi ngắn -> tiếp tục
        if current_streak <= avg_streak:
            cont = decode(s[-1])
            conf = min(50 + (avg_streak - current_streak) * 10, 85)
            return cont, conf, f"Streak {current_streak} <= avg {avg_streak:.1f}"
        
        return None, 0, "Streak trung lập"

# ---------- 5. REVERSAL DETECTION ----------
class ReversalModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        
        s = encode(tx_list)
        
        # Đếm số lần đảo trong 15 phiên gần nhất
        recent = s[-15:]
        reversals = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        reversal_rate = reversals / (len(recent) - 1)
        
        # Tỷ lệ đảo cao -> tiếp tục đảo
        if reversal_rate > 0.6:
            next_c = "X" if recent[-1] == "T" else "T"
            return decode(next_c), min(50 + reversal_rate * 30, 85), f"Đảo cao {reversal_rate:.0%}"
        
        # Tỷ lệ đảo thấp -> tiếp tục trend
        if reversal_rate < 0.3:
            return decode(recent[-1]), min(50 + (0.3-reversal_rate)*100, 80), f"Ít đảo {reversal_rate:.0%}"
        
        return None, 0, f"Đảo trung bình {reversal_rate:.0%}"

# ---------- 6. CYCLE PREDICTION ----------
class CycleModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        
        s = encode(tx_list)
        
        # Tìm chu kỳ bằng autocorrelation đơn giản
        best_cycle = None
        best_score = 0
        
        for cycle in range(2, min(11, len(s)//2)):
            matches = 0
            total = 0
            for i in range(cycle, len(s)):
                if s[i] == s[i-cycle]:
                    matches += 1
                total += 1
            
            score = matches / total if total > 0 else 0
            if score > best_score and score > 0.55:
                best_score = score
                best_cycle = cycle
        
        if best_cycle:
            predicted = s[-best_cycle]
            conf = min(score * 100, 90)
            return decode(predicted), conf, f"Chu kỳ {best_cycle} ({best_score:.0%})"
        
        return None, 0, "Không có chu kỳ rõ ràng"

# ---------- 7. MOMENTUM MODEL ----------
class MomentumModel:
    def predict(self, tx_list):
        if len(tx_list) < 10:
            return None, 0
        
        s = encode(tx_list)
        
        # Tính momentum theo các cửa sổ
        windows = [5, 10, 15]
        momentums = []
        
        for w in windows:
            if len(s) >= w:
                recent = s[-w:]
                t_pct = recent.count("T") / w
                momentums.append(t_pct)
        
        if not momentums:
            return None, 0
        
        avg_momentum = sum(momentums) / len(momentums)
        
        # Momentum > 0.6 -> Tài mạnh
        if avg_momentum > 0.6:
            return "Tài", min(avg_momentum * 100, 90), f"Momentum Tài {avg_momentum:.0%}"
        elif avg_momentum < 0.4:
            return "Xỉu", min((1-avg_momentum) * 100, 90), f"Momentum Xỉu {1-avg_momentum:.0%}"
        
        return None, 0, f"Momentum trung lập {avg_momentum:.0%}"

# ---------- 8. TREND STRENGTH ----------
class TrendStrengthModel:
    def predict(self, tx_list):
        if len(tx_list) < 15:
            return None, 0
        
        s = encode(tx_list)
        
        # So sánh 2 nửa
        half = len(s) // 2
        first_half = s[:half].count("T") / half if half > 0 else 0.5
        second_half = s[half:].count("T") / (len(s)-half) if len(s)-half > 0 else 0.5
        
        trend_diff = abs(second_half - first_half)
        
        if trend_diff > 0.15:
            # Trend mạnh -> tiếp tục
            if second_half > first_half:
                return "Tài", min(50 + trend_diff * 200, 85), f"Trend Tài mạnh"
            else:
                return "Xỉu", min(50 + trend_diff * 200, 85), f"Trend Xỉu mạnh"
        
        return None, 0, f"Trend yếu {trend_diff:.0%}"

# ---------- 9. MOVING AVERAGE ----------
class MovingAverageModel:
    def predict(self, tx_list):
        if len(tx_list) < 20:
            return None, 0
        
        # Chuyển Tài=1, Xỉu=0 rồi tính MA
        values = [1 if x == "Tài" else 0 for x in tx_list]
        
        ma5 = sum(values[-5:]) / 5
        ma10 = sum(values[-10:]) / 10
        ma20 = sum(values[-20:]) / 20 if len(values) >= 20 else ma10
        
        # Golden cross / Death cross đơn giản
        if ma5 > ma10 > ma20:
            return "Tài", 75, "MA Bullish"
        elif ma5 < ma10 < ma20:
            return "Xỉu", 75, "MA Bearish"
        
        # MA5 cắt lên
        if len(values) >= 6:
            prev_ma5 = sum(values[-6:-1]) / 5
            if ma5 > prev_ma5 and ma5 > 0.5:
                return "Tài", 70, "MA5 tăng"
            elif ma5 < prev_ma5 and ma5 < 0.5:
                return "Xỉu", 70, "MA5 giảm"
        
        return None, 0, f"MA5={ma5:.2f} MA10={ma10:.2f}"

# ---------- 10. ENSEMBLE LEARNING ----------
class EnsembleModel:
    def __init__(self):
        self.models = {
            "Markov": MarkovChain(),
            "N-Gram": NGramModel(),
            "Pattern": PatternModel(),
            "Streak": StreakModel(),
            "Reversal": ReversalModel(),
            "Cycle": CycleModel(),
            "Momentum": MomentumModel(),
            "Trend": TrendStrengthModel(),
            "MA": MovingAverageModel()
        }
    
    def predict(self, tx_list):
        votes = Counter()
        confidences = {}
        reasons = {}
        
        for name, model in self.models.items():
            try:
                pred, conf, reason = model.predict(tx_list)
                if pred:
                    votes[pred] += conf  # Weighted voting
                    confidences[name] = round(conf, 1)
                    reasons[name] = reason
            except:
                continue
        
        if not votes:
            # Fallback: đảo cầu
            if tx_list:
                fallback = "Xỉu" if tx_list[-1] == "Tài" else "Tài"
                return fallback, 50, {"Fallback": "Đảo cầu"}, "Không có tín hiệu - đảo"
            return "Tài", 50, {}, "Mặc định"
        
        # Chọn winner
        winner = votes.most_common(1)[0]
        pred = winner[0]
        total_conf = winner[1]
        avg_conf = min(total_conf / sum(votes.values()) * 100, 95) if sum(votes.values()) > 0 else 50
        
        return pred, round(avg_conf, 1), confidences, " | ".join([f"{k}:{v}" for k,v in reasons.items()])

# ---------- 11. ADAPTIVE WEIGHT ----------
class AdaptiveWeightModel:
    def __init__(self):
        self.model = EnsembleModel()
        self.weights = {name: 1.0 for name in self.model.models.keys()}
        self.performance = {name: {"dung": 0, "sai": 0} for name in self.model.models.keys()}
    
    def update_weights(self, prediction, actual, model_outputs):
        """Điều chỉnh trọng số dựa trên kết quả thực tế"""
        for name, pred in model_outputs.items():
            if pred == actual:
                self.performance[name]["dung"] += 1
                self.weights[name] = min(self.weights[name] * 1.05, 3.0)
            else:
                self.performance[name]["sai"] += 1
                self.weights[name] = max(self.weights[name] * 0.95, 0.3)
    
    def predict(self, tx_list):
        pred, conf, model_confs, reason = self.model.predict(tx_list)
        return pred, conf, model_confs, reason, self.weights.copy()

# Khởi tạo AI
ai_engine = AdaptiveWeightModel()

# ===============================
# FETCH DATA
# ===============================

def fetch_loop():
    global last_phien, last_prediction, stats, latest_data
    
    while True:
        try:
            res = requests.get(API_URL, timeout=5)
            data = res.json()
            ds = data.get("list", [])
            if not ds:
                time.sleep(2)
                continue
            
            item = ds[0]
            phien = item.get("id")
            
            if phien == last_phien:
                time.sleep(2)
                continue
            
            d1, d2, d3 = item.get("dices")
            tong = item.get("point")
            ket = "Tài" if tong >= 11 else "Xỉu"
            
            # Lưu lịch sử
            history_tx.append(ket)
            history_pt.append(tong)
            history_id.append(phien)
            history_dice.append((d1, d2, d3))
            
            # Tính chuỗi
            s = encode(list(history_tx))
            max_tai = max_xiu = 0
            cur_tai = cur_xiu = 0
            for c in s:
                if c == "T":
                    cur_tai += 1
                    cur_xiu = 0
                    max_tai = max(max_tai, cur_tai)
                else:
                    cur_xiu += 1
                    cur_tai = 0
                    max_xiu = max(max_xiu, cur_xiu)
            
            stats["max_tai"] = max_tai
            stats["max_xiu"] = max_xiu
            
            # Tính chuỗi hiện tại
            current_streak = 0
            current_type = None
            if s:
                current_type = decode(s[-1])
                for c in reversed(s):
                    if (c == "T" and current_type == "Tài") or (c == "X" and current_type == "Xỉu"):
                        current_streak += 1
                    else:
                        break
            
            # Kiểm tra dự đoán trước
            if last_prediction is not None and len(history_tx) > 1:
                stats["tong"] += 1
                correct = (last_prediction == ket)
                if correct:
                    stats["dung"] += 1
                else:
                    stats["sai"] += 1
                
                pred_log.append({
                    "phien": phien,
                    "prediction": last_prediction,
                    "actual": ket,
                    "correct": correct
                })
            
            # AI DỰ ĐOÁN
            tx_list = list(history_tx)
            if len(tx_list) >= MIN_PHIEN:
                du_doan, do_tin_cay, model_confs, phan_tich, weights = ai_engine.predict(tx_list)
            else:
                du_doan = "Xỉu" if ket == "Tài" else "Tài"  # Đảo cầu ban đầu
                do_tin_cay = 50
                model_confs = {}
                phan_tich = f"Chờ thêm {MIN_PHIEN - len(tx_list)} phiên"
                weights = {}
            
            last_prediction = du_doan
            
            # Pattern
            pattern = s[-20:] if len(s) >= 20 else s
            
            # Tỷ lệ thật
            ty_le = round(stats["dung"] / max(stats["tong"], 1) * 100, 1) if stats["tong"] > 0 else 0
            
            latest_data = {
                "Phiên": phien,
                "Xúc xắc 1": d1,
                "Xúc xắc 2": d2,
                "Xúc xắc 3": d3,
                "Tổng": tong,
                "Kết": ket,
                "Phiên hiện tại": phien + 1,
                "Dự đoán": du_doan,
                "Độ tin cậy": do_tin_cay,
                "Pattern": pattern,
                "Max chuỗi Tài": max_tai,
                "Max chuỗi Xỉu": max_xiu,
                "AI Models": model_confs,
                "Trọng số": {k: round(v, 2) for k, v in weights.items()},
                "Phân tích": phan_tich,
                "ID": "tuananh"
            }
            
            
            
            last_phien = phien
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi: {e}")
        
        time.sleep(2)

# ===============================
# API
# ===============================

@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify(latest_data)

@app.route("/api/lichsu", methods=["GET"])
def api_lichsu():
    ls = []
    for i in range(-min(20, len(history_id)), 0):
        idx = len(history_id) + i
        ls.append({
            "phiên": history_id[idx],
            "kết_quả": history_tx[idx],
            "tổng": history_pt[idx]
        })
    
    recent_pred = list(pred_log)[-20:]
    
    return jsonify({
        "tổng_quan": {
            "đã_quan_sát": stats["tong"],
            "đúng": stats["dung"],
            "sai": stats["sai"],
            "tỷ_lệ": latest_data.get("Tỷ lệ đúng", "0%"),
            "max_tai": stats["max_tai"],
            "max_xiu": stats["max_xiu"]
        },
        "lịch_sử_dự_đoán": recent_pred,
        "lịch_sử_20_phiên": ls
    })

# ===============================
# RUN
# ===============================
threading.Thread(target=fetch_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
