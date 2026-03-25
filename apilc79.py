import requests
import time
import threading
import statistics
from flask import Flask, jsonify
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Optional

# ===============================
# CẤU HÌNH
# ===============================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
last_processed_session_id = None

app = Flask(__name__)

# ===============================
# DỮ LIỆU TOÀN CỤC
# ===============================
history = []  # Lưu Tài/Xỉu
history_details = []  # Lưu chi tiết đầy đủ
MAX_HISTORY = 100
MIN_PHIEN_PREDICT = 10

latest_data = {
    "Phiên": None,
    "Xúc xắc 1": None,
    "Xúc xắc 2": None,
    "Xúc xắc 3": None,
    "Tổng": None,
    "Kết": None,
    "Phiên hiện tại": None,
    "Dự đoán": "Đang chờ",
    "Độ tin cậy": 0.0,
    "ID": "tuananh"
}

# Database học sâu
pattern_db = defaultdict(lambda: {'T': 0, 'X': 0, 'total': 0})
streak_stats = {'Tài': [], 'Xỉu': []}
transition_count = {'TT': 0, 'TX': 0, 'XT': 0, 'XX': 0}
accuracy_log = []

# ===============================
# 10 THUẬT TOÁN AI THÔNG MINH
# ===============================

class TaiXiuAI:
    def __init__(self):
        self.weights = {
            'ngram_deep': 0.15,
            'markov_chain': 0.12,
            'streak_predictor': 0.12,
            'frequency_regression': 0.11,
            'entropy_analyzer': 0.10,
            'cycle_detector': 0.10,
            'sequence_matcher': 0.10,
            'momentum_calculator': 0.08,
            'alternating_detector': 0.07,
            'bayesian_fusion': 0.05
        }
    
    # THUẬT TOÁN 1: N-GRAM DEEP LEARNING
    def algo_ngram_deep(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 8:
            return None, 0, "Không đủ dữ liệu"
        
        best_conf = 0
        best_pred = None
        best_n = 0
        
        for n in range(3, 8):
            if len(data) < n + 1:
                continue
            
            current = tuple(data[-n:])
            matches = {'T': 0, 'X': 0, 'total': 0}
            
            for i in range(len(data) - n - 1):
                past = tuple(data[i:i+n])
                if past == current:
                    next_val = 'T' if data[i+n] == 'Tài' else 'X'
                    weight = (i + 1) / len(data)
                    matches[next_val] += weight
                    matches['total'] += weight
            
            if matches['total'] >= 2:
                t_prob = matches['T'] / matches['total']
                x_prob = matches['X'] / matches['total']
                
                if abs(t_prob - x_prob) > 0.2:
                    pred = 'Tài' if t_prob > x_prob else 'Xỉu'
                    conf = 50 + abs(t_prob - x_prob) * 50
                    if conf > best_conf:
                        best_conf = conf
                        best_pred = pred
                        best_n = n
        
        if best_pred:
            return best_pred, min(best_conf, 90), f"N-gram cấp {best_n}"
        return None, 0, "Không tìm thấy pattern khớp"
    
    # THUẬT TOÁN 2: MARKOV CHAIN NÂNG CAO
    def algo_markov_chain(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 5:
            return None, 0, "Không đủ dữ liệu"
        
        # Cập nhật ma trận
        for i in range(len(data) - 1):
            curr = 'T' if data[i] == 'Tài' else 'X'
            next_val = 'T' if data[i+1] == 'Tài' else 'X'
            transition_count[curr + next_val] += 1
        
        last = 'T' if data[-1] == 'Tài' else 'X'
        t_next = transition_count[last + 'T']
        x_next = transition_count[last + 'X']
        total = t_next + x_next
        
        if total < 5:
            return None, 0, "Không đủ mẫu chuyển tiếp"
        
        t_prob = t_next / total
        x_prob = x_next / total
        
        if abs(t_prob - x_prob) < 0.15:
            return None, 0, "Xác suất quá cân bằng"
        
        pred = 'Tài' if t_prob > x_prob else 'Xỉu'
        conf = 50 + abs(t_prob - x_prob) * 50
        
        return pred, conf, f"Markov từ {last}"
    
    # THUẬT TOÁN 3: STREAK PREDICTOR
    def algo_streak_predictor(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 10:
            return None, 0, "Không đủ dữ liệu"
        
        last = data[-1]
        current_streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                current_streak += 1
            else:
                break
        
        # Thu thập thống kê chuỗi
        all_streaks = []
        curr_streak = 1
        curr_type = data[0]
        
        for i in range(1, len(data)):
            if data[i] == curr_type:
                curr_streak += 1
            else:
                all_streaks.append((curr_type, curr_streak))
                curr_type = data[i]
                curr_streak = 1
        all_streaks.append((curr_type, curr_streak))
        
        same_type = [s[1] for s in all_streaks if s[0] == last]
        if not same_type:
            return None, 0, "Không có dữ liệu chuỗi"
        
        avg = statistics.mean(same_type)
        max_s = max(same_type)
        
        if current_streak >= max_s - 1 and max_s >= 4:
            return 'Xỉu' if last == 'Tài' else 'Tài', 80, f"Chuỗi {current_streak} gần max {max_s}"
        elif current_streak > avg + 1:
            return 'Xỉu' if last == 'Tài' else 'Tài', 65, f"Chuỗi {current_streak} > TB {avg:.1f}"
        elif current_streak == 1:
            return last, 55, "Chuỗi mới bắt đầu"
        
        return None, 0, f"Chuỗi {current_streak} bình thường"
    
    # THUẬT TOÁN 4: FREQUENCY REGRESSION
    def algo_frequency_regression(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 20:
            return None, 0, "Không đủ dữ liệu"
        
        windows = [5, 10, 20]
        ratios = []
        
        for w in windows:
            recent = data[-w:]
            t_ratio = recent.count('Tài') / w
            ratios.append((w, t_ratio))
        
        short = ratios[0][1]
        long = ratios[-1][1]
        trend = short - long
        
        if abs(trend) > 0.2:
            pred = 'Tài' if trend > 0 else 'Xỉu'
            conf = 55 + abs(trend) * 100
            return pred, min(conf, 85), f"Xu hướng mạnh {trend:+.0%}"
        
        avg_ratio = statistics.mean([r[1] for r in ratios])
        if avg_ratio > 0.6:
            return 'Xỉu', 60, f"Tần suất Tài cao {avg_ratio:.0%}"
        elif avg_ratio < 0.4:
            return 'Tài', 60, f"Tần suất Xỉu cao {1-avg_ratio:.0%}"
        
        return None, 0, "Tần suất cân bằng"
    
    # THUẬT TOÁN 5: ENTROPY ANALYZER
    def algo_entropy_analyzer(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        window = data[-15:]
        switches = sum(1 for i in range(len(window)-1) if window[i] != window[i+1])
        entropy = switches / 14
        
        hist_entropies = []
        for i in range(len(data) - 15):
            w = data[i:i+15]
            s = sum(1 for j in range(len(w)-1) if w[j] != w[j+1])
            hist_entropies.append(s / 14)
        
        if not hist_entropies:
            return None, 0, "Không có lịch sử"
        
        avg_e = statistics.mean(hist_entropies)
        std_e = statistics.stdev(hist_entropies) if len(hist_entropies) > 1 else 0.1
        
        z = (entropy - avg_e) / std_e if std_e > 0 else 0
        
        if z < -1.5:
            return window[-1], 75, f"Entropy thấp (z={z:.2f}), tiếp tục"
        elif z > 1.5:
            return 'Xỉu' if window[-1] == 'Tài' else 'Tài', 70, f"Entropy cao (z={z:.2f}), đảo"
        
        return None, 0, f"Entropy bình thường"
    
    # THUẬT TOÁN 6: CYCLE DETECTOR
    def algo_cycle_detector(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 24:
            return None, 0, "Không đủ dữ liệu"
        
        series = [1 if x == 'Tài' else 0 for x in data[-40:]]
        
        best_cycle = None
        best_score = 0
        
        for cycle in range(2, 13):
            if len(series) < cycle * 3:
                continue
            
            matches = 0
            total = 0
            for i in range(0, len(series) - cycle * 2, cycle):
                for j in range(cycle):
                    if series[i+j] == series[i+cycle+j]:
                        matches += 1
                    total += 1
            
            if total > 0:
                score = matches / total
                if score > best_score and score > 0.65:
                    best_score = score
                    best_cycle = cycle
        
        if best_cycle:
            pos = len(series) % best_cycle
            pos = best_cycle if pos == 0 else pos
            
            historical = []
            for i in range(best_cycle, len(series), best_cycle):
                idx = i - best_cycle + pos - 1
                if 0 <= idx < len(series):
                    historical.append(series[idx])
            
            if historical:
                prob = statistics.mean(historical)
                pred = 'Tài' if prob > 0.5 else 'Xỉu'
                conf = 50 + abs(prob - 0.5) * 100
                return pred, min(conf, 80), f"Chu kỳ {best_cycle} khớp {best_score:.0%}"
        
        return None, 0, "Không phát hiện chu kỳ"
    
    # THUẬT TOÁN 7: SEQUENCE MATCHER
    def algo_sequence_matcher(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 12:
            return None, 0, "Không đủ dữ liệu"
        
        current = ''.join(['T' if x == 'Tài' else 'X' for x in data[-8:]])
        
        best_sim = 0
        best_next = None
        
        for i in range(len(data) - 9):
            past = ''.join(['T' if x == 'Tài' else 'X' for x in data[i:i+8]])
            next_val = data[i+8]
            
            matches = sum(a == b for a, b in zip(current, past))
            sim = matches / 8
            
            if sim > best_sim and sim >= 0.75:
                best_sim = sim
                best_next = next_val
        
        if best_next:
            conf = 50 + (best_sim - 0.5) * 100
            return best_next, min(conf, 85), f"Giống {best_sim:.0%} pattern cũ"
        
        return None, 0, "Không tìm thấy pattern tương tự"
    
    # THUẬT TOÁN 8: MOMENTUM CALCULATOR
    def algo_momentum_calculator(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 16:
            return None, 0, "Không đủ dữ liệu"
        
        n = len(data) // 4
        phases = [data[i*n:(i+1)*n] for i in range(4) if (i+1)*n <= len(data)]
        
        if len(phases) < 4:
            return None, 0, "Không đủ pha"
        
        ratios = [p.count('Tài') / len(p) for p in phases]
        momentums = [ratios[i+1] - ratios[i] for i in range(3)]
        
        if abs(momentums[-1]) > 0.15:
            pred = 'Tài' if momentums[-1] > 0 else 'Xỉu'
            conf = 55 + abs(momentums[-1]) * 100
            return pred, min(conf, 80), f"Momentum mạnh"
        
        if len(momentums) >= 2:
            acc = momentums[-1] - momentums[-2]
            if abs(acc) > 0.1:
                pred = 'Xỉu' if momentums[-1] > 0 else 'Tài'
                return pred, 60, f"Gia tốc đảo chiều"
        
        return None, 0, "Momentum yếu"
    
    # THUẬT TOÁN 9: ALTERNATING DETECTOR
    def algo_alternating_detector(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 6:
            return None, 0, "Không đủ dữ liệu"
        
        recent = data[-6:]
        alternating = all(recent[i] != recent[i+1] for i in range(5))
        
        if alternating:
            # Đang T-X-T-X-T-X, khả năng cao phá vỡ hoặc tiếp tục
            if data[-2] == data[-1]:
                return data[-1], 65, "Pattern T-X đồng nhất"
            else:
                return 'Xỉu' if data[-1] == 'Tài' else 'Tài', 60, "Pattern T-X chuẩn"
        
        # Kiểm tra gần alternating
        switches = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        if switches >= 4:
            return recent[-1], 55, "Gần alternating"
        
        return None, 0, "Không có pattern alternating"
    
    # THUẬT TOÁN 10: BAYESIAN FUSION
    def algo_bayesian_fusion(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 10:
            return None, 0, "Không đủ dữ liệu"
        
        log_odds = 0
        
        # Evidence 1: Ngắn hạn
        r5 = data[-5:]
        t5 = r5.count('Tài') / 5
        if t5 > 0.7:
            log_odds -= 0.6
        elif t5 < 0.3:
            log_odds += 0.6
        
        # Evidence 2: Trung hạn
        if len(data) >= 15:
            r15 = data[-15:]
            t15 = r15.count('Tài') / 15
            log_odds -= (t15 - 0.5) * 1.5
        
        # Evidence 3: Chuỗi
        last = data[-1]
        streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                streak += 1
            else:
                break
        
        if streak >= 3:
            log_odds += -0.4 * streak if last == 'Tài' else 0.4 * streak
        
        prob = 1 / (1 + 2.71828 ** (-log_odds))
        
        if 0.45 <= prob <= 0.55:
            return None, 0, f"Bayes không chắc chắn"
        
        pred = 'Tài' if prob > 0.5 else 'Xỉu'
        conf = 50 + abs(prob - 0.5) * 100
        
        return pred, conf, f"Bayes {prob:.0%}"
    
    # ENSEMBLE VOTING
    def predict(self, data: List[str]) -> Tuple[str, float, str, list]:
        if len(data) < MIN_PHIEN_PREDICT:
            return "Chờ dữ liệu", 0, f"Cần {MIN_PHIEN_PREDICT} phiên", []
        
        algorithms = [
            ('N-Gram Deep', self.algo_ngram_deep),
            ('Markov Chain', self.algo_markov_chain),
            ('Streak AI', self.algo_streak_predictor),
            ('Frequency Reg', self.algo_frequency_regression),
            ('Entropy', self.algo_entropy_analyzer),
            ('Cycle Detect', self.algo_cycle_detector),
            ('Seq Matcher', self.algo_sequence_matcher),
            ('Momentum', self.algo_momentum_calculator),
            ('Alternating', self.algo_alternating_detector),
            ('Bayesian', self.algo_bayesian_fusion)
        ]
        
        votes = {'Tài': 0.0, 'Xỉu': 0.0}
        details = []
        
        for name, algo in algorithms:
            try:
                result, conf, reason = algo(data)
                if result and conf >= 50:
                    weight = self.weights[name.lower().replace(' ', '_')]
                    vote_power = weight * (conf / 100)
                    votes[result] += vote_power
                    
                    details.append({
                        'name': name,
                        'prediction': result,
                        'confidence': round(conf, 1),
                        'reason': reason[:40]
                    })
            except:
                continue
        
        total = votes['Tài'] + votes['Xỉu']
        if total == 0:
            return "Không chắc chắn", 0, "Không đủ tín hiệu", details
        
        t_ratio = votes['Tài'] / total
        final = 'Tài' if t_ratio > 0.5 else 'Xỉu'
        margin = abs(t_ratio - 0.5)
        conf = 50 + margin * 100
        
        # Điều chỉnh theo số AI đồng thuận
        same = [d for d in details if d['prediction'] == final]
        conf += len(same) * 2
        conf = min(conf, 95)
        
        reason = f"{len(same)} AI đồng thuận" if same else "Tổng hợp"
        
        return final, round(conf, 1), reason, details


# Khởi tạo AI
ai_engine = TaiXiuAI()


# ===============================
# BOT LẤY DỮ LIỆU
# ===============================
def fetch_data_loop():
    global last_processed_session_id, latest_data, history, history_details, accuracy_log

    last_prediction = None
    last_conf = 0

    print("=" * 70)
    print("🧠 SUPER AI TÀI XỈU - 10 THUẬT TOÁN SIÊU TỐI ƯU")
    print("=" * 70)
    print("⏳ Đang chờ dữ liệu từ API...")
    print("-" * 70)

    while True:
        try:
            res = requests.get(API_URL, timeout=10)
            data = res.json()

            list_data = data.get("list", [])
            if not list_data:
                time.sleep(2)
                continue

            phien = list_data[0]
            phien_id = phien.get("id")

            if phien_id == last_processed_session_id:
                time.sleep(2)
                continue

            dices = phien.get("dices")
            tong = phien.get("point")
            d1, d2, d3 = dices
            
            ket_qua = "Tài" if tong >= 11 else "Xỉu"
            last_processed_session_id = phien_id

            # Lưu lịch sử
            history.append(ket_qua)
            history_details.append({
                'phien': phien_id,
                'dices': dices,
                'tong': tong,
                'ket_qua': ket_qua,
                'time': datetime.now().strftime("%H:%M:%S")
            })
            
            if len(history) > MAX_HISTORY:
                history.pop(0)
                history_details.pop(0)

            # Kiểm tra dự đoán cũ
            if last_prediction and ket_qua in ['Tài', 'Xỉu']:
                correct = (last_prediction == ket_qua)
                accuracy_log.append(correct)
                status = "✅ ĐÚNG" if correct else "❌ SAI"
                acc_20 = sum(accuracy_log[-20:]) / min(len(accuracy_log), 20) * 100 if accuracy_log else 0
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 PHIÊN {phien_id}")
                print(f"    🎲 Xúc xắc: {d1}-{d2}-{d3} = {tong} điểm → [{ket_qua}]")
                print(f"    🎯 Dự đoán trước: {last_prediction} ({last_conf}%) → {status}")
                print(f"    📈 Độ chính xác 20 phiên: {acc_20:.1f}%")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🆕 PHIÊN MỚI {phien_id}")
                print(f"    🎲 Xúc xắc: {d1}-{d2}-{d3} = {tong} điểm → [{ket_qua}]")

            # Dự đoán mới
            if len(history) >= MIN_PHIEN_PREDICT:
                du_doan, do_tin_cay, ly_do, chi_tiet = ai_engine.predict(history)
                last_prediction = du_doan if du_doan in ['Tài', 'Xỉu'] else None
                last_conf = do_tin_cay
                
                # Kiểm tra đảo cầu
                pattern_str = ''.join(['T' if x == 'Tài' else 'X' for x in history[-10:]])
                
                print(f"    🔮 DỰ ĐOÁN TIẾP THEO: {du_doan} (độ tin: {do_tin_cay}%)")
                print(f"       Lý do: {ly_do}")
                print(f"       Pattern 10: {pattern_str}")
                
                if chi_tiet:
                    top3 = sorted(chi_tiet, key=lambda x: x['confidence'], reverse=True)[:3]
                    print(f"       Top AI: " + " | ".join([f"{t['name']}({t['confidence']:.0f}%)" for t in top3]))
            else:
                print(f"    ⏳ Chờ thêm {MIN_PHIEN_PREDICT - len(history)} phiên...")
                last_prediction = None
            
            

            # Cập nhật JSON
            latest_data.update({
                "Phiên": phien_id,
                "Xúc xắc 1": d1,
                "Xúc xắc 2": d2,
                "Xúc xắc 3": d3,
                "Tổng": tong,
                "Kết": ket_qua,
                "Phiên hiện tại": phien_id,
                "Dự đoán": last_prediction if last_prediction else "Chờ",
                "Độ tin cậy": last_conf,
                "ID": "tuananh"
            })

        except Exception as e:
            print(f"[LỖI] {str(e)[:50]}")

        time.sleep(2)


# ===============================
# CHẠY THREAD
# ===============================
threading.Thread(target=fetch_data_loop, daemon=True).start()


# ===============================
# API
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify({"data": latest_data})


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({
        "history": history_details[-50:],
        "pattern": ''.join(['T' if x == 'Tài' else 'X' for x in history]),
        "accuracy": {
            "total": len(accuracy_log),
            "correct": sum(accuracy_log),
            "rate": round(sum(accuracy_log) / len(accuracy_log) * 100, 2) if accuracy_log else 0
        }
    })


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    print("Server đang chạy...")
    app.run(host="0.0.0.0", port=10000)
