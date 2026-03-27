import requests
import time
import threading
import statistics
import math
from flask import Flask, jsonify
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# ===============================
# CẤU HÌNH
# ===============================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
last_processed_session_id = None

app = Flask(__name__)

# ===============================
# DỮ LIỆU TOÀN CỤC
# ===============================
history = []
history_details = []
MAX_HISTORY = 200
MIN_PHIEN_PREDICT = 20

latest_data = {}

# Theo dõi bias
bias_tracker = {'Tài': 0, 'Xỉu': 0, 'corrections': 0}
accuracy_tracker = {'correct': 0, 'total': 0, 'tai_correct': 0, 'xiu_correct': 0}

# ===============================
# 20 THUẬT TOÁN AI CÔNG BẰNG
# ===============================

class BalancedAI:
    def __init__(self):
        raw_weights = {
            'pattern_20': 0.08,
            'ngram_5': 0.07,
            'ngram_7': 0.07,
            'markov_2': 0.06,
            'markov_3': 0.06,
            'streak_deep': 0.06,
            'frequency_5': 0.05,
            'frequency_10': 0.05,
            'frequency_20': 0.05,
            'frequency_50': 0.05,
            'entropy_15': 0.05,
            'entropy_30': 0.05,
            'cycle_detect': 0.05,
            'sequence_match': 0.05,
            'momentum': 0.05,
            'trend_analysis': 0.05,
            'alternating': 0.05,
            'bayesian': 0.05,
            'golden_ratio': 0.04,
            'chaos_theory': 0.04,
            'balance_guard': 0.08
        }

        # Normalize để tổng = 1
        total = sum(raw_weights.values())
        self.weights = {k: v / total for k, v in raw_weights.items()}

    def to_tx(self, result: str) -> str:
        return 'T' if result == 'Tài' else 'X'

    def to_full(self, tx: str) -> str:
        return 'Tài' if tx == 'T' else 'Xỉu'
    
    def check_bias(self, data: List[str]) -> float:
        """Kiểm tra độ lệch của dữ liệu"""
        if len(data) < 20:
            return 0.0
        t_ratio = data.count('Tài') / len(data)
        return t_ratio - 0.5  # >0 lệch Tài, <0 lệch Xỉu
    
    # ========== THUẬT TOÁN 1: PATTERN 20 CÂN BẰNG ==========
    def algo_pattern_20(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 21:
            return None, 0, "Không đủ dữ liệu"
        
        current_20 = ''.join([self.to_tx(x) for x in data[-20:]])
        
        # Đếm T và X trong pattern hiện tại
        t_count = current_20.count('T')
        x_count = current_20.count('X')
        
        # Nếu đã lệch quá nhiều -> đảo
        if t_count >= 14:  # 70% T
            return 'Xỉu', 70, f"Pattern 20 lệch T ({t_count}/20)"
        if x_count >= 14:  # 70% X
            return 'Tài', 70, f"Pattern 20 lệch X ({x_count}/20)"
        
        # Tìm pattern tương tự trong lịch sử
        matches = {'T': 0, 'X': 0, 'total': 0}
        for i in range(len(data) - 21):
            past_20 = ''.join([self.to_tx(x) for x in data[i:i+20]])
            if past_20 == current_20:
                next_val = data[i+20]
                matches[self.to_tx(next_val)] += 1
                matches['total'] += 1
        
        if matches['total'] >= 2:
            t_prob = matches['T'] / matches['total']
            # Chỉ dự đoán nếu chênh lệch rõ ràng
            if t_prob > 0.6:
                return 'Tài', 60, f"Pattern 20 khớp T ({matches['total']} lần)"
            elif t_prob < 0.4:
                return 'Xỉu', 60, f"Pattern 20 khớp X ({matches['total']} lần)"
        
        return None, 0, "Pattern 20 trung lập"
    
    # ========== THUẬT TOÁN 2-3: N-GRAM CÂN BẰNG ==========
    def algo_ngram(self, data: List[str], n: int) -> Tuple[Optional[str], float, str]:
        if len(data) < n + 1:
            return None, 0, f"Không đủ N-{n}"
        
        current = tuple(data[-n:])
        matches = {'T': 0, 'X': 0, 'total': 0}
        
        for i in range(len(data) - n - 1):
            if tuple(data[i:i+n]) == current:
                next_val = 'T' if data[i+n] == 'Tài' else 'X'
                matches[next_val] += 1
                matches['total'] += 1
        
        if matches['total'] >= 3:
            t_prob = matches['T'] / matches['total']
            # Ngưỡng cao hơn để tránh dự đoán lung tung
            if t_prob > 0.65:
                return 'Tài', 65, f"N-{n}: T mạnh"
            elif t_prob < 0.35:
                return 'Xỉu', 65, f"N-{n}: X mạnh"
        
        return None, 0, f"N-{n} không rõ"
    
    # ========== THUẬT TOÁN 4-5: MARKOV CÂN BẰNG ==========
    def algo_markov(self, data: List[str], order: int) -> Tuple[Optional[str], float, str]:
        if len(data) < order + 1:
            return None, 0, f"Không đủ Markov-{order}"
        
        # Xây dựng ma trận
        transitions = defaultdict(lambda: {'T': 0, 'X': 0})
        
        for i in range(len(data) - order):
            key = ''.join([self.to_tx(data[i+j]) for j in range(order)])
            next_val = self.to_tx(data[i+order])
            transitions[key][next_val] += 1
        
        current_key = ''.join([self.to_tx(data[-order+j]) for j in range(order)])
        
        if current_key not in transitions:
            return None, 0, f"Chưa thấy Markov-{order} này"
        
        trans = transitions[current_key]
        total = trans['T'] + trans['X']
        
        if total < 5:
            return None, 0, f"Markov-{order} ít mẫu"
        
        t_prob = trans['T'] / total
        
        # Ngưỡng chặt chẽ hơn
        if t_prob > 0.7:
            return 'Tài', 65, f"Markov-{order}: T"
        elif t_prob < 0.3:
            return 'Xỉu', 65, f"Markov-{order}: X"
        
        return None, 0, f"Markov-{order} cân bằng"
    
    # ========== THUẬT TOÁN 6: STREAK CÂN BẰNG ==========
    def algo_streak(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        last = data[-1]
        current_streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                current_streak += 1
            else:
                break
        
        # Thu thập chuỗi
        all_streaks = []
        curr_type = data[0]
        curr_count = 1
        
        for i in range(1, len(data)):
            if data[i] == curr_type:
                curr_count += 1
            else:
                all_streaks.append((curr_type, curr_count))
                curr_type = data[i]
                curr_count = 1
        all_streaks.append((curr_type, curr_count))
        
        same_type = [s[1] for s in all_streaks if s[0] == last]
        if not same_type:
            return None, 0, "Không có dữ liệu chuỗi"
        
        avg = statistics.mean(same_type)
        max_s = max(same_type)
        
        # Chỉ đảo khi chuỗi thực sự dài
        if current_streak >= 5:  # Chuỗi 5+ mới đảo
            return 'Xỉu' if last == 'Tài' else 'Tài', 75, f"Chuỗi {current_streak} rất dài"
        elif current_streak == 4 and current_streak >= max_s - 1:
            return 'Xỉu' if last == 'Tài' else 'Tài', 65, f"Chuỗi 4 gần max"
        
        return None, 0, f"Chuỗi {current_streak} bình thường"
    
    # ========== THUẬT TOÁN 7-10: FREQUENCY CÂN BẰNG ==========
    def algo_frequency(self, data: List[str], window: int) -> Tuple[Optional[str], float, str]:
        if len(data) < window:
            return None, 0, f"Không đủ {window} phiên"
        
        recent = data[-window:]
        t_ratio = recent.count('Tài') / window
        
        # Ngưỡng cao hơn, không nhạy cảm
        if t_ratio > 0.75:  # 75% mới đảo
            return 'Xỉu', 70, f"Freq-{window}: T quá nhiều ({t_ratio:.0%})"
        elif t_ratio < 0.25:
            return 'Tài', 70, f"Freq-{window}: X quá nhiều ({t_ratio:.0%})"
        
        return None, 0, f"Freq-{window} cân bằng"
    
    # ========== THUẬT TOÁN 11-12: ENTROPY CÂN BẰNG ==========
    def algo_entropy(self, data: List[str], window: int) -> Tuple[Optional[str], float, str]:
        if len(data) < window + 5:
            return None, 0, f"Không đủ entropy-{window}"
        
        recent = data[-window:]
        switches = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        entropy = switches / (window - 1)
        
        # Entropy thấp = chuỗi dài -> tiếp tục
        if entropy < 0.15:  # Rất thấp
            return recent[-1], 70, f"Entropy-{window} rất thấp, tiếp tục"
        elif entropy > 0.85:  # Rất cao
            return 'Xỉu' if recent[-1] == 'Tài' else 'Tài', 65, f"Entropy-{window} rất cao, đảo"
        
        return None, 0, f"Entropy-{window} bình thường"
    
    # ========== THUẬT TOÁN 13: CYCLE CÂN BẰNG ==========
    def algo_cycle(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 30:
            return None, 0, "Không đủ dữ liệu"
        
        series = [1 if x == 'Tài' else 0 for x in data[-50:]]
        
        best_cycle = None
        best_score = 0
        
        for cycle in range(2, 15):
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
                if score > best_score and score > 0.8:  # Khắt khe hơn
                    best_score = score
                    best_cycle = cycle
        
        if best_cycle and best_score > 0.8:
            pos = len(series) % best_cycle
            pos = best_cycle if pos == 0 else pos
            
            historical = []
            for i in range(best_cycle, len(series), best_cycle):
                idx = i - best_cycle + pos - 1
                if 0 <= idx < len(series):
                    historical.append(series[idx])
            
            if historical:
                prob = statistics.mean(historical)
                if abs(prob - 0.5) > 0.2:  # Chỉ khi rõ ràng
                    pred = 'Tài' if prob > 0.5 else 'Xỉu'
                    conf = 60 + abs(prob - 0.5) * 80
                    return pred, min(conf, 85), f"Chu kỳ {best_cycle} rõ ràng"
        
        return None, 0, "Không phát hiện chu kỳ rõ"
    
    # ========== THUẬT TOÁN 14: SEQUENCE MATCH CÂN BẰNG ==========
    def algo_sequence(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        for length in [12, 15, 10]:  # Thử nhiều độ dài
            if len(data) < length + 1:
                continue
            
            current = ''.join([self.to_tx(x) for x in data[-length:]])
            
            best_sim = 0
            best_next = None
            
            for i in range(len(data) - length - 1):
                past = ''.join([self.to_tx(x) for x in data[i:i+length]])
                matches = sum(a == b for a, b in zip(current, past))
                sim = matches / length
                
                if sim > best_sim and sim >= 0.85:  # Khắt khe hơn
                    best_sim = sim
                    best_next = data[i+length]
            
            if best_next and best_sim >= 0.85:
                return best_next, 70, f"Match {length} ký tự ({best_sim:.0%})"
        
        return None, 0, "Không tìm thấy match đủ tốt"
    
    # ========== THUẬT TOÁN 15: MOMENTUM CÂN BẰNG ==========
    def algo_momentum(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 20:
            return None, 0, "Không đủ dữ liệu"
        
        n = len(data) // 4
        phases = [data[i*n:(i+1)*n] for i in range(4)]
        
        ratios = [p.count('Tài') / len(p) for p in phases]
        
        # Tính momentum
        mom3 = ratios[3] - ratios[2]
        
        # Chỉ khi momentum rất mạnh
        if mom3 > 0.25:
            return 'Tài', 65, f"Momentum T mạnh"
        elif mom3 < -0.25:
            return 'Xỉu', 65, f"Momentum X mạnh"
        
        return None, 0, "Momentum trung lập"
    
    # ========== THUẬT TOÁN 16: TREND CÂN BẰNG ==========
    def algo_trend(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 25:
            return None, 0, "Không đủ dữ liệu"
        
        recent = data[-20:]
        y = [1 if x == 'Tài' else 0 for x in recent]
        x = list(range(len(y)))
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Chỉ khi slope rõ ràng
        if slope > 0.08:
            return 'Tài', 60, f"Trend T rõ"
        elif slope < -0.08:
            return 'Xỉu', 60, f"Trend X rõ"
        
        return None, 0, f"Trend không rõ"
    
    # ========== THUẬT TOÁN 17: ALTERNATING CÂN BẰNG ==========
    def algo_alternating(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 8:
            return None, 0, "Không đủ dữ liệu"
        
        recent = data[-8:]
        
        # Kiểm tra alternating hoàn hảo
        perfect = all(recent[i] != recent[i+1] for i in range(7))
        
        if perfect:
            # Pattern T-X-T-X-T-X-T-X
            if recent[-1] == 'Tài':
                return 'Xỉu', 65, "Alternating: tiếp X"
            else:
                return 'Tài', 65, "Alternating: tiếp T"
        
        # Kiểm tra gần alternating
        switches = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        if switches == 7:  # Gần hoàn hảo
            return recent[-1], 55, "Gần alternating"
        
        return None, 0, "Không alternating"
    
    # ========== THUẬT TOÁN 18: BAYESIAN CÂN BẰNG ==========
    def algo_bayesian(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        # Prior cân bằng 50-50
        log_odds = 0
        
        # Evidence ngắn hạn
        short = data[-5:]
        t_short = short.count('Tài') / 5
        
        # Chỉ khi cực đoan
        if t_short > 0.9:
            log_odds -= 1.5
        elif t_short < 0.1:
            log_odds += 1.5
        
        # Evidence chuỗi
        last = data[-1]
        streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                streak += 1
            else:
                break
        
        if streak >= 4:
            log_odds += -0.8 * streak if last == 'Tài' else 0.8 * streak
        
        prob = 1 / (1 + math.exp(-log_odds))
        
        # Chỉ dự đoán khi rõ ràng
        if prob > 0.7:
            return 'Tài', 65, f"Bayes T rõ ({prob:.0%})"
        elif prob < 0.3:
            return 'Xỉu', 65, f"Bayes X rõ ({prob:.0%})"
        
        return None, 0, f"Bayes không rõ"
    
    # ========== THUẬT TOÁN 19: GOLDEN CÂN BẰNG ==========
    def algo_golden(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 30:
            return None, 0, "Không đủ dữ liệu"
        
        cut = int(len(data) * 0.618)
        
        first = data[:cut]
        second = data[cut:]
        
        t_first = first.count('Tài') / len(first)
        t_second = second.count('Tài') / len(second)
        
        diff = abs(t_second - t_first)
        
        # Chỉ khi chênh lệch lớn
        if diff > 0.25:
            pred = 'Xỉu' if t_second > t_first else 'Tài'
            return pred, 60, f"Golden lệch {diff:.0%}"
        
        return None, 0, "Golden cân bằng"
    
    # ========== THUẬT TOÁN 20: CHAOS CÂN BẰNG ==========
    def algo_chaos(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 20:
            return None, 0, "Không đủ dữ liệu"
        
        # Tìm điểm 5 giống nhau
        for i in range(5, len(data)):
            window = data[i-5:i]
            if len(set(window)) == 1:
                # Điểm nhạy cảm
                next_vals = []
                for j in range(i, min(i+5, len(data))):
                    next_vals.append(data[j])
                
                if next_vals:
                    t_count = next_vals.count('Tài')
                    if t_count >= 4:
                        return 'Tài', 65, f"Chaos -> T"
                    elif t_count <= 1:
                        return 'Xỉu', 65, f"Chaos -> X"
        
        return None, 0, "Không ở điểm nhạy cảm"
    
    # ========== BẢO VỆ CÂN BẰNG ==========
    def balance_guard(self, data: List[str], current_prediction: Optional[str]) -> Tuple[Optional[str], float, str]:
        """Kiểm tra và điều chỉnh nếu bị lệch"""
        global bias_tracker
        
        if len(data) < 30:
            return current_prediction, 0, "Chưa đủ dữ liệu kiểm tra"
        
        # Tính tỷ lệ dự đoán gần đây
        recent_10 = data[-10:]
        t_ratio = recent_10.count('Tài') / 10
        
        # Nếu đang lệch nhiều về một bên
        if t_ratio > 0.8:  # 8/10 Tài
            bias_tracker['Tài'] += 1
            return 'Xỉu', 75, "Bảo vệ cân bằng: quá nhiều T"
        elif t_ratio < 0.2:  # 2/10 Tài
            bias_tracker['Xỉu'] += 1
            return 'Tài', 75, "Bảo vệ cân bằng: quá nhiều X"
        
        return current_prediction, 0, "Cân bằng tốt"
    
    # ========== ENSEMBLE VOTING CÂN BẰNG ==========
    def predict(self, data: List[str]) -> Tuple[str, float, str, List[Dict], str]:
        """Dự đoán với kiểm tra cân bằng"""
        if len(data) < MIN_PHIEN_PREDICT:
            pattern_20 = ''.join([self.to_tx(x) for x in data]) if data else ""
            return "Chờ dữ liệu", 0, f"Cần {MIN_PHIEN_PREDICT} phiên", [], pattern_20
        
        algorithms = [
            ('Pattern 20', self.algo_pattern_20),
         ('N-Gram 5', lambda d: self.algo_ngram(d, 5)),
            ('N-Gram 7', lambda d: self.algo_ngram(d, 7)),
            ('Markov 2', lambda d: self.algo_markov(d, 2)),
            ('Markov 3', lambda d: self.algo_markov(d, 3)),
            ('Streak', self.algo_streak),
            ('Freq 5', lambda d: self.algo_frequency(d, 5)),
            ('Freq 10', lambda d: self.algo_frequency(d, 10)),
            ('Freq 20', lambda d: self.algo_frequency(d, 20)),
            ('Freq 50', lambda d: self.algo_frequency(d, 50)),
            ('Entropy 15', lambda d: self.algo_entropy(d, 15)),
            ('Entropy 30', lambda d: self.algo_entropy(d, 30)),
            ('Cycle', self.algo_cycle),
            ('Sequence', self.algo_sequence),
            ('Momentum', self.algo_momentum),
            ('Trend', self.algo_trend),
            ('Alternating', self.algo_alternating),
            ('Bayesian', self.algo_bayesian),
            ('Golden', self.algo_golden),
            ('Chaos', self.algo_chaos)
        ]
        
        
        # phần này nằm trong class BalancedAI

def predict(self, data):
    votes = {'Tài': 0.0, 'Xỉu': 0.0}
    details = []

    # ===== chạy các algo trước đó để fill votes + details =====
    for name, algo in self.algorithms:
        try:
            result, conf, reason = algo(data)

            if result and conf is not None and conf >= 55:
                key = name.lower().replace(' ', '_').replace('-', '_')
                weight = self.weights.get(key, 0)

                vote_power = weight * (conf / 100)
                votes[result] += vote_power

                details.append({
                    'name': name,
                    'prediction': result,
                    'confidence': round(conf, 1),
                    'weight': weight,
                    'vote': round(vote_power, 3),
                    'reason': reason
                })

        except Exception as e:
            print(f"Lỗi ở {name}: {e}")
            continue

    # ===== pattern 20 =====
    pattern_20 = ''.join([self.to_tx(x) for x in data[-20:]])

    # ===== guard =====
    total_votes = votes['Tài'] + votes['Xỉu']

    if total_votes == 0:
        guard_pred, guard_conf, guard_reason = self.balance_guard(data, None)

        if guard_pred in ['Tài', 'Xỉu']:
            return guard_pred, guard_conf, guard_reason, [], pattern_20

        return "Không chắc chắn", 0, "Không đủ tín hiệu", [], pattern_20

    # ===== tính kết quả =====
    t_ratio = votes['Tài'] / total_votes
    final = 'Tài' if t_ratio > 0.5 else 'Xỉu'
    margin = abs(t_ratio - 0.5)

    # ===== balance guard =====
    guard_pred, guard_conf, guard_reason = self.balance_guard(data, final)

    if guard_pred in ['Tài', 'Xỉu'] and margin < 0.15:
        final = guard_pred
        reason = guard_reason
        conf = guard_conf
    else:
        base_conf = 50 + margin * 100
        active = len([d for d in details if d['prediction'] == final])
        conf = min(base_conf + active * 1.5, 95)

        same = [d for d in details if d['prediction'] == final]
        if same:
            top = max(same, key=lambda x: x['vote'])
            reason = f"{top['name']} mạnh, {active}/20 đồng ý"
        else:
            reason = "Tổng hợp"

    return final, round(conf, 1), reason, details, pattern_20
        
       
    def predict(self, data):   # ✅ indent 4 spaces trong class
        total_votes = votes['Tài'] + votes['Xỉu']

        if total_votes > 0:
            t_ratio = votes['Tài'] / total_votes
        else:
            t_ratio = 0.5

        final = 'Tài' if t_ratio > 0.5 else 'Xỉu'
        margin = abs(t_ratio - 0.5)

        return final, margin
        
        # Kiểm tra cân bằng
        guard_pred, guard_conf, guard_reason = self.balance_guard(data, final)
        
        # Nếu balance guard can thiệp và margin nhỏ, nghe theo guard
        if guard_pred and margin < 0.15:
            final = guard_pred
            reason = guard_reason
            conf = guard_conf
        else:
            # Tính confidence
            base_conf = 50 + margin * 100
            active = len([d for d in details if d['prediction'] == final])
            conf = min(base_conf + active * 1.5, 95)
            
            same = [d for d in details if d['prediction'] == final]
            if same:
                top = max(same, key=lambda x: x['vote'])
                reason = f"{top['name']} mạnh, {active}/20 đồng ý"
            else:
                reason = "Tổng hợp"
        
        return final, round(conf, 1), reason, details, pattern_20


# Khởi tạo AI
balanced_ai = BalancedAI()


# ===============================
# BOT LẤY DỮ LIỆU
# ===============================
def fetch_data_loop():
    global last_processed_session_id, latest_data, history, history_details, accuracy_tracker, bias_tracker

    last_prediction = None
    last_conf = 0

    

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
                accuracy_tracker['history'].append(correct)
                accuracy_tracker['total'] += 1
                if correct:
                    accuracy_tracker['correct'] += 1
                    if ket_qua == 'Tài':
                        accuracy_tracker['tai_correct'] += 1
                    else:
                        accuracy_tracker['xiu_correct'] += 1
                
                status = "✅ ĐÚNG" if correct else "❌ SAI"
                acc_rate = accuracy_tracker['correct'] / accuracy_tracker['total'] * 100 if accuracy_tracker['total'] > 0 else 0
                
                # Kiểm tra bias
                tai_rate = history.count('Tài') / len(history) if history else 0.5
                
                

            # Dự đoán mới
            du_doan, do_tin_cay, ly_do, chi_tiet, pattern_20 = balanced_ai.predict(history)
            
            if du_doan in ['Tài', 'Xỉu']:
                last_prediction = du_doan
                last_conf = do_tin_cay
            else:
                last_prediction = None

            
            
            latest_data = {
                "phiên": phien_id,
                "xúc_xắc_1": d1,
                "xúc_xắc_2": d2,
                "xúc_xắc_3": d3,
                "tổng": tong,
                "kết": ket_qua,
                "dự_đoán": du_doan if du_doan in ['Tài', 'Xỉu'] else "Chờ",
                "pattern": pattern_20,
                "độ_tin_cậy": do_tin_cay,
                "id": "tuananh"
            }

        except Exception as e:
            print(f"[LỖI] {str(e)[:60]}")

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
    pattern_all = ''.join(['T' if x == 'Tài' else 'X' for x in history])
    tai_count = history.count('Tài')
    xiu_count = history.count('Xỉu')
    
    return jsonify({
        "history": history_details[-50:],
        "pattern_full": pattern_all,
        "pattern_20_gần_nhất": pattern_all[-20:] if len(pattern_all) >= 20 else pattern_all,
        "số_phiên_đã_lưu": len(history),
        "tỷ_lệ_tài_trong_lịch_sử": round(tai_count / len(history) * 100, 1) if history else 50,
        "tỷ_lệ_xỉu_trong_lịch_sử": round(xiu_count / len(history) * 100, 1) if history else 50,
        "độ_chính_xác": {
            "tổng": accuracy_tracker['total'],
            "đúng": accuracy_tracker['correct'],
            "tỷ_lệ": round(accuracy_tracker['correct'] / accuracy_tracker['total'] * 100, 1) if accuracy_tracker['total'] > 0 else 0,
            "tài_đúng": accuracy_tracker['tai_correct'],
            "xỉu_đúng": accuracy_tracker['xiu_correct']
        },
        "cân_bằng_can_thiệp": bias_tracker
    })





# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    print("Server đang chạy trên port 10000...")
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)
