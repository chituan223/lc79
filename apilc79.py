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
history = []  # Lưu Tài/Xỉu
history_details = []  # Lưu chi tiết
MAX_HISTORY = 200
MIN_PHIEN_PREDICT = 20  # Tối thiểu 20 phiên mới dự đoán

latest_data = {}

# Database học sâu
pattern_20_db = defaultdict(lambda: {'T': 0, 'X': 0, 'total': 0})  # Pattern 20 ký tự
all_patterns = {}  # Lưu tất cả pattern đã thấy
transition_matrix = defaultdict(lambda: defaultdict(int))
streak_history = {'Tài': [], 'Xỉu': []}
window_stats = {}
accuracy_tracker = {'correct': 0, 'total': 0, 'history': deque(maxlen=50)}

# ===============================
# 20 THUẬT TOÁN AI THÔNG MINH
# ===============================

class SuperAI:
    def __init__(self):
        self.weights = {
            'pattern_20': 0.10,      # Pattern 20 ký tự
            'ngram_5': 0.08,         # N-gram cấp 5
            'ngram_7': 0.08,         # N-gram cấp 7
            'markov_2': 0.07,        # Markov 2 bước
            'markov_3': 0.07,        # Markov 3 bước
            'streak_deep': 0.07,     # Phân tích chuỗi sâu
            'frequency_5': 0.05,     # Tần suất 5 phiên
            'frequency_10': 0.05,    # Tần suất 10 phiên
            'frequency_20': 0.05,    # Tần suất 20 phiên
            'frequency_50': 0.05,    # Tần suất 50 phiên
            'entropy_15': 0.05,      # Entropy 15 phiên
            'entropy_30': 0.05,      # Entropy 30 phiên
            'cycle_detect': 0.05,    # Phát hiện chu kỳ
            'sequence_match': 0.05,  # So khớp chuỗi
            'momentum': 0.04,        # Động lượng
            'trend_analysis': 0.04,  # Phân tích xu hướng
            'alternating': 0.03,     # Pattern T-X-T-X
            'bayesian': 0.03,        # Bayes tổng hợp
            'golden_ratio': 0.02,    # Tỷ lệ vàng
            'chaos_theory': 0.02     # Lý thuyết hỗn loạn
        }
    
    def to_tx(self, result: str) -> str:
        return 'T' if result == 'Tài' else 'X'
    
    def from_tx(self, tx: str) -> str:
        return 'Tài' if tx == 'T' else 'Xỉu'
    
    # ========== THUẬT TOÁN 1: PATTERN 20 KÝ TỰ ==========
    def algo_pattern_20(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        """Phân tích pattern 20 ký tự chi tiết"""
        if len(data) < 21:
            return None, 0, "Không đủ dữ liệu"
        
        current_20 = ''.join([self.to_tx(x) for x in data[-20:]])
        
        # Tìm trong lịch sử
        matches = {'T': 0, 'X': 0, 'total': 0}
        for i in range(len(data) - 21):
            past_20 = ''.join([self.to_tx(x) for x in data[i:i+20]])
            if past_20 == current_20:
                next_val = data[i+20]
                weight = (i + 1) / len(data)  # Trọng số thời gian
                matches[self.to_tx(next_val)] += weight
                matches['total'] += weight
        
        if matches['total'] >= 2:
            t_prob = matches['T'] / matches['total']
            x_prob = matches['X'] / matches['total']
            if abs(t_prob - x_prob) > 0.15:
                pred = 'Tài' if t_prob > x_prob else 'Xỉu'
                conf = 50 + abs(t_prob - x_prob) * 50
                return pred, min(conf, 95), f"Pattern 20 khớp {matches['total']:.1f} lần"
        
        # Phân tích con pattern trong 20 ký tự
        sub_patterns = {
            'TTTTT': 0, 'XXXXX': 0,
            'TXTXTX': 0, 'XTXTXT': 0,
            'TTTXX': 0, 'XXXTT': 0
        }
        
        for pattern in sub_patterns.keys():
            sub_patterns[pattern] = current_20.count(pattern)
        
        # Nếu có chuỗi dài -> đảo
        if sub_patterns['TTTTT'] >= 1:
            return 'Xỉu', 65, "Có chuỗi TTTTT trong 20"
        if sub_patterns['XXXXX'] >= 1:
            return 'Tài', 65, "Có chuỗi XXXXX trong 20"
        
        return None, 0, "Pattern 20 không rõ ràng"
    
    # ========== THUẬT TOÁN 2: N-GRAM CẤP 5 ==========
    def algo_ngram_5(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 6:
            return None, 0, "Không đủ dữ liệu"
        
        current = tuple(data[-5:])
        matches = {'T': 0, 'X': 0, 'total': 0}
        
        for i in range(len(data) - 6):
            if tuple(data[i:i+5]) == current:
                next_val = 'T' if data[i+5] == 'Tài' else 'X'
                matches[next_val] += 1
                matches['total'] += 1
        
        if matches['total'] >= 3:
            t_prob = matches['T'] / matches['total']
            if abs(t_prob - 0.5) > 0.2:
                pred = 'Tài' if t_prob > 0.5 else 'Xỉu'
                conf = 50 + abs(t_prob - 0.5) * 100
                return pred, min(conf, 85), f"N-gram 5: {matches['total']} mẫu"
        
        return None, 0, "N-gram 5 không đủ mẫu"
    
    # ========== THUẬT TOÁN 3: N-GRAM CẤP 7 ==========
    def algo_ngram_7(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 8:
            return None, 0, "Không đủ dữ liệu"
        
        current = tuple(data[-7:])
        matches = {'T': 0, 'X': 0, 'total': 0}
        
        for i in range(len(data) - 8):
            if tuple(data[i:i+7]) == current:
                next_val = 'T' if data[i+7] == 'Tài' else 'X'
                matches[next_val] += 1
                matches['total'] += 1
        
        if matches['total'] >= 2:
            t_prob = matches['T'] / matches['total']
            if abs(t_prob - 0.5) > 0.25:
                pred = 'Tài' if t_prob > 0.5 else 'Xỉu'
                conf = 55 + abs(t_prob - 0.5) * 90
                return pred, min(conf, 90), f"N-gram 7: {matches['total']} mẫu"
        
        return None, 0, "N-gram 7 không đủ mẫu"
    
    # ========== THUẬT TOÁN 4: MARKOV 2 BƯỚC ==========
    def algo_markov_2(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 3:
            return None, 0, "Không đủ dữ liệu"
        
        # Cập nhật ma trận
        for i in range(len(data) - 1):
            curr = self.to_tx(data[i])
            next_val = self.to_tx(data[i+1])
            transition_matrix[curr][next_val] += 1
        
        last = self.to_tx(data[-1])
        t_count = transition_matrix[last]['T']
        x_count = transition_matrix[last]['X']
        total = t_count + x_count
        
        if total < 10:
            return None, 0, "Markov 2 chưa đủ dữ liệu"
        
        t_prob = t_count / total
        if abs(t_prob - 0.5) > 0.1:
            pred = 'Tài' if t_prob > 0.5 else 'Xỉu'
            conf = 50 + abs(t_prob - 0.5) * 100
            return pred, conf, f"Markov 2 từ {last}"
        
        return None, 0, "Markov 2 cân bằng"
    
    # ========== THUẬT TOÁN 5: MARKOV 3 BƯỚC ==========
    def algo_markov_3(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 4:
            return None, 0, "Không đủ dữ liệu"
        
        # Ma trận 3 bước: TT->?, TX->?, XT->?, XX->?
        markov_3 = defaultdict(lambda: {'T': 0, 'X': 0})
        
        for i in range(len(data) - 2):
            key = self.to_tx(data[i]) + self.to_tx(data[i+1])
            next_val = self.to_tx(data[i+2])
            markov_3[key][next_val] += 1
        
        last2 = self.to_tx(data[-2]) + self.to_tx(data[-1])
        if last2 not in markov_3:
            return None, 0, "Chưa thấy pattern 2 này"
        
        trans = markov_3[last2]
        total = trans['T'] + trans['X']
        
        if total < 5:
            return None, 0, "Markov 3 không đủ mẫu"
        
        t_prob = trans['T'] / total
        if abs(t_prob - 0.5) > 0.15:
            pred = 'Tài' if t_prob > 0.5 else 'Xỉu'
            conf = 50 + abs(t_prob - 0.5) * 100
            return pred, min(conf, 85), f"Markov 3 từ {last2}"
        
        return None, 0, "Markov 3 cân bằng"
    
    # ========== THUẬT TOÁN 6: PHÂN TÍCH CHUỖI SÂU ==========
    def algo_streak_deep(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        last = data[-1]
        current_streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                current_streak += 1
            else:
                break
        
        # Thu thập tất cả chuỗi
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
            return None, 0, "Không có dữ liệu"
        
        avg = statistics.mean(same_type)
        std = statistics.stdev(same_type) if len(same_type) > 1 else 0
        max_s = max(same_type)
        
        # Z-score của chuỗi hiện tại
        z_score = (current_streak - avg) / std if std > 0 else 0
        
        if current_streak >= max_s - 1 and max_s >= 4:
            return 'Xỉu' if last == 'Tài' else 'Tài', 85, f"Chuỗi {current_streak} gần max ({max_s})"
        elif z_score > 1.5:
            return 'Xỉu' if last == 'Tài' else 'Tài', 75, f"Chuỗi cao (z={z_score:.2f})"
        elif current_streak > avg + 1.5:
            return 'Xỉu' if last == 'Tài' else 'Tài', 65, f"Chuỗi > TB + 1.5"
        elif current_streak == 1:
            return last, 55, "Bắt đầu chuỗi mới"
        
        return None, 0, f"Chuỗi {current_streak} bình thường (z={z_score:.2f})"
    
    # ========== THUẬT TOÁN 7-10: TẦN SUẤT ĐA CỬA SỔ ==========
    def algo_frequency(self, data: List[str], window: int) -> Tuple[Optional[str], float, str]:
        if len(data) < window:
            return None, 0, f"Không đủ {window} phiên"
        
        recent = data[-window:]
        t_count = recent.count('Tài')
        t_ratio = t_count / window
        
        deviation = abs(t_ratio - 0.5)
        
        if window == 5 and deviation > 0.3:
            pred = 'Xỉu' if t_ratio > 0.5 else 'Tài'
            return pred, 60, f"Nhiễu 5 phiên ({t_ratio:.0%})"
        elif window == 10 and deviation > 0.25:
            pred = 'Xỉu' if t_ratio > 0.5 else 'Tài'
            return pred, 65, f"Lệch 10 phiên ({t_ratio:.0%})"
        elif window == 20 and deviation > 0.2:
            pred = 'Xỉu' if t_ratio > 0.5 else 'Tài'
            return pred, 70, f"Hồi quy 20 phiên ({t_ratio:.0%})"
        elif window == 50 and deviation > 0.15:
            pred = 'Xỉu' if t_ratio > 0.5 else 'Tài'
            return pred, 75, f"Hồi quy 50 phiên ({t_ratio:.0%})"
        
        return None, 0, f"Tần suất {window} cân bằng"
    
    # ========== THUẬT TOÁN 11-12: ENTROPY ==========
    def algo_entropy(self, data: List[str], window: int) -> Tuple[Optional[str], float, str]:
        if len(data) < window + 5:
            return None, 0, f"Không đủ dữ liệu entropy {window}"
        
        recent = data[-window:]
        switches = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        entropy = switches / (window - 1)
        
        # Tính entropy lịch sử
        hist_entropies = []
        for i in range(len(data) - window):
            w = data[i:i+window]
            s = sum(1 for j in range(len(w)-1) if w[j] != w[j+1])
            hist_entropies.append(s / (window - 1))
        
        if not hist_entropies:
            return None, 0, "Không có lịch sử entropy"
        
        avg_e = statistics.mean(hist_entropies)
        std_e = statistics.stdev(hist_entropies) if len(hist_entropies) > 1 else 0.05
        
        z = (entropy - avg_e) / std_e if std_e > 0 else 0
        
        if z < -1.3:  # Entropy thấp -> chuỗi dài
            return recent[-1], 70, f"Entropy thấp {window} (z={z:.2f})"
        elif z > 1.3:  # Entropy cao -> hỗn loạn, sắp đổi
            return 'Xỉu' if recent[-1] == 'Tài' else 'Tài', 65, f"Entropy cao {window} (z={z:.2f})"
        
        return None, 0, f"Entropy {window} bình thường"
    
    # ========== THUẬT TOÁN 13: PHÁT HIỆN CHU KỲ ==========
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
                if score > best_score and score > 0.7:
                    best_score = score
                    best_cycle = cycle
        
        if best_cycle and best_score > 0.75:
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
                return pred, min(conf, 88), f"Chu kỳ {best_cycle} (độ khớp {best_score:.0%})"
        
        return None, 0, "Không phát hiện chu kỳ rõ ràng"
    
    # ========== THUẬT TOÁN 14: SO KHỚP CHUỖI ==========
    def algo_sequence_match(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        # Test các độ dài pattern
        for length in [10, 12, 15]:
            if len(data) < length + 1:
                continue
            
            current = ''.join([self.to_tx(x) for x in data[-length:]])
            
            best_sim = 0
            best_next = None
            
            for i in range(len(data) - length - 1):
                past = ''.join([self.to_tx(x) for x in data[i:i+length]])
                matches = sum(a == b for a, b in zip(current, past))
                sim = matches / length
                
                if sim > best_sim and sim >= 0.8:
                    best_sim = sim
                    best_next = data[i+length]
            
            if best_next and best_sim >= 0.8:
                conf = 50 + (best_sim - 0.5) * 100
                return best_next, min(conf, 90), f"Giống {length} ký tự ({best_sim:.0%})"
        
        return None, 0, "Không tìm thấy chuỗi tương tự"
    
    # ========== THUẬT TOÁN 15: ĐỘNG LƯỢNG ==========
    def algo_momentum(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 20:
            return None, 0, "Không đủ dữ liệu"
        
        # Chia 4 pha
        n = len(data) // 4
        phases = [data[i*n:(i+1)*n] for i in range(4)]
        
        ratios = [p.count('Tài') / len(p) for p in phases]
        
        # Tính momentum
        mom1 = ratios[1] - ratios[0]
        mom2 = ratios[2] - ratios[1]
        mom3 = ratios[3] - ratios[2]
        
        acceleration = mom3 - mom2
        
        if abs(mom3) > 0.2:
            pred = 'Tài' if mom3 > 0 else 'Xỉu'
            return pred, 65, f"Momentum mạnh ({mom3:+.0%})"
        
        if abs(acceleration) > 0.15:
            pred = 'Xỉu' if mom3 > 0 else 'Tài'
            return pred, 60, f"Gia tốc đảo ({acceleration:+.0%})"
        
        return None, 0, "Momentum ổn định"
    
    # ========== THUẬT TOÁN 16: XU HƯỚNG ==========
    def algo_trend(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 25:
            return None, 0, "Không đủ dữ liệu"
        
        # Linear regression đơn giản
        recent = data[-20:]
        y = [1 if x == 'Tài' else 0 for x in recent]
        x = list(range(len(y)))
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        if slope > 0.05:
            return 'Tài', 60, f"Xu hướng tăng (slope={slope:.3f})"
        elif slope < -0.05:
            return 'Xỉu', 60, f"Xu hướng giảm (slope={slope:.3f})"
        
        return None, 0, f"Xu hướng ngang (slope={slope:.3f})"
    
    # ========== THUẬT TOÁN 17: ALTERNATING ==========
    def algo_alternating(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 8:
            return None, 0, "Không đủ dữ liệu"
        
        recent = data[-8:]
        
        # Kiểm tra pattern T-X-T-X
        perfect_alt = all(recent[i] != recent[i+1] for i in range(7))
        
        if perfect_alt:
            if recent[-2] == recent[-1]:
                return recent[-1], 70, "Alternating hoàn hảo, tiếp tục"
            else:
                return 'Xỉu' if recent[-1] == 'Tài' else 'Tài', 65, "Alternating chuẩn, đảo"
        
        # Gần alternating
        switches = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])
        if switches >= 6:
            return recent[-1], 55, f"Gần alternating ({switches}/7)"
        
        return None, 0, "Không có pattern alternating"
    
    # ========== THUẬT TOÁN 18: BAYESIAN ==========
    def algo_bayesian(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 15:
            return None, 0, "Không đủ dữ liệu"
        
        log_odds = 0
        
        # Prior từ tần suất dài hạn
        long_term = data[-50:] if len(data) >= 50 else data
        t_ratio = long_term.count('Tài') / len(long_term)
        log_odds += (0.5 - t_ratio) * 2  # Hồi quy về 0.5
        
        # Evidence ngắn hạn
        short = data[-5:]
        t_short = short.count('Tài') / 5
        if t_short > 0.8:
            log_odds -= 1.0
        elif t_short < 0.2:
            log_odds += 1.0
        
        # Evidence chuỗi
        last = data[-1]
        streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                streak += 1
            else:
                break
        
        if streak >= 3:
            log_odds += -0.5 * streak if last == 'Tài' else 0.5 * streak
        
        prob = 1 / (1 + math.exp(-log_odds))
        
        if 0.4 <= prob <= 0.6:
            return None, 0, f"Bayes không chắc chắn ({prob:.0%})"
        
        pred = 'Tài' if prob > 0.5 else 'Xỉu'
        conf = 50 + abs(prob - 0.5) * 100
        
        return pred, conf, f"Bayes {prob:.0%}"
    
    # ========== THUẬT TOÁN 19: TỶ LỆ VÀNG ==========
    def algo_golden(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 30:
            return None, 0, "Không đủ dữ liệu"
        
        phi = 0.618
        cut_point = int(len(data) * phi)
        
        if cut_point < 10:
            return None, 0, "Không đủ dữ liệu Fibonacci"
        
        first_part = data[:cut_point]
        second_part = data[cut_point:]
        
        t_first = first_part.count('Tài') / len(first_part)
        t_second = second_part.count('Tài') / len(second_part)
        
        # Nếu phần sau lệch nhiều so với phần đầu -> đảo
        diff = t_second - t_first
        
        if abs(diff) > 0.2:
            pred = 'Xỉu' if t_second > 0.5 else 'Tài'
            return pred, 60, f"Golden cut lệch {diff:+.0%}"
        
        return None, 0, "Golden cut cân bằng"
    
    # ========== THUẬT TOÁN 20: HỖN LOẠN ==========
    def algo_chaos(self, data: List[str]) -> Tuple[Optional[str], float, str]:
        if len(data) < 20:
            return None, 0, "Không đủ dữ liệu"
        
        # Tìm điểm bifurcation (nơi hệ thống đổi trạng thái)
        changes = []
        for i in range(5, len(data)-1):
            # Pattern trước khi đổi
            before = data[i-5:i]
            if len(set(before)) == 1:  # 5 giống nhau
                changes.append((before[0], data[i]))
        
        if len(changes) < 3:
            return None, 0, "Không đủ điểm bifurcation"
        
        # Kiểm tra pattern hiện tại
        current_5 = data[-5:]
        if len(set(current_5)) == 1:
            # Đang ở điểm nhạy cảm
            same_type = [c[1] for c in changes if c[0] == current_5[0]]
            if same_type:
                dem = Counter(same_type)
                pred = dem.most_common(1)[0][0]
                prob = dem[pred] / len(same_type)
                conf = 50 + prob * 40
                return pred, conf, f"Chaos point ({len(same_type)} mẫu)"
        
        return None, 0, "Không ở điểm nhạy cảm"
    
    # ========== ENSEMBLE VOTING 20 AI ==========
    def predict(self, data: List[str]) -> Tuple[str, float, str, List[Dict], str]:
        """Kết hợp 20 thuật toán"""
        if len(data) < MIN_PHIEN_PREDICT:
            pattern_20 = ''.join([self.to_tx(x) for x in data]) if data else ""
            return "Chờ dữ liệu", 0, f"Cần {MIN_PHIEN_PREDICT} phiên", [], pattern_20
        
        algorithms = [
            ('Pattern 20', self.algo_pattern_20),
            ('N-Gram 5', self.algo_ngram_5),
            ('N-Gram 7', self.algo_ngram_7),
            ('Markov 2', self.algo_markov_2),
            ('Markov 3', self.algo_markov_3),
            ('Streak Deep', self.algo_streak_deep),
            ('Freq 5', lambda d: self.algo_frequency(d, 5)),
            ('Freq 10', lambda d: self.algo_frequency(d, 10)),
            ('Freq 20', lambda d: self.algo_frequency(d, 20)),
            ('Freq 50', lambda d: self.algo_frequency(d, 50)),
            ('Entropy 15', lambda d: self.algo_entropy(d, 15)),
            ('Entropy 30', lambda d: self.algo_entropy(d, 30)),
            ('Cycle', self.algo_cycle),
            ('Seq Match', self.algo_sequence_match),
            ('Momentum', self.algo_momentum),
            ('Trend', self.algo_trend),
            ('Alternating', self.algo_alternating),
            ('Bayesian', self.algo_bayesian),
            ('Golden', self.algo_golden),
            ('Chaos', self.algo_chaos)
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
                        'weight': weight,
                        'vote': round(vote_power, 3),
                        'reason': reason
                    })
            except Exception as e:
                continue
        
        # Tạo pattern 20 ký tự
        pattern_20 = ''.join([self.to_tx(x) for x in data[-20:]])
        
        total = votes['Tài'] + votes['Xỉu']
        if total == 0:
            return "Không chắc chắn", 0, "Không đủ tín hiệu", details, pattern_20
        
        t_ratio = votes['Tài'] / total
        final = 'Tài' if t_ratio > 0.5 else 'Xỉu'
        margin = abs(t_ratio - 0.5)
        
        # Tính confidence thật
        base_conf = 50 + margin * 100
        active_algos = len([d for d in details if d['prediction'] == final])
        conf_bonus = active_algos * 1.5
        final_conf = min(base_conf + conf_bonus, 98)
        
        # Lý do tổng hợp
        same_pred = [d for d in details if d['prediction'] == final]
        if same_pred:
            top = max(same_pred, key=lambda x: x['vote'])
            reason = f"{top['name']} mạnh nhất, {active_algos}/20 AI đồng ý"
        else:
            reason = "Tổng hợp 20 thuật toán"
        
        return final, round(final_conf, 1), reason, details, pattern_20


# Khởi tạo AI
super_ai = SuperAI()


# ===============================
# BOT LẤY DỮ LIỆU
# ===============================
def fetch_data_loop():
    global last_processed_session_id, latest_data, history, history_details, accuracy_tracker

    last_prediction = None
    last_conf = 0

    print("=" * 80)
    print("🧠 SUPER AI 2.0 - 20 THUẬT TOÁN + PATTERN 20 KÝ TỰ")
    print("=" * 80)
    print("⏳ Đang chờ dữ liệu từ API...")
    print("-" * 80)

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
                
                status = "✅ ĐÚNG" if correct else "❌ SAI"
                acc_rate = accuracy_tracker['correct'] / accuracy_tracker['total'] * 100 if accuracy_tracker['total'] > 0 else 0
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 PHIÊN {phien_id}")
                print(f"    🎲 Xúc xắc: [{d1}] [{d2}] [{d3}] = {tong} điểm → [{ket_qua}]")
                print(f"    🎯 Dự đoán trước: {last_prediction} ({last_conf}%) → {status}")
                print(f"    📈 Tỷ lệ chính xác: {acc_rate:.1f}% ({accuracy_tracker['correct']}/{accuracy_tracker['total']})")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🆕 PHIÊN MỚI {phien_id}")
                print(f"    🎲 Xúc xắc: [{d1}] [{d2}] [{d3}] = {tong} điểm → [{ket_qua}]")

            # Dự đoán mới với 20 AI
            du_doan, do_tin_cay, ly_do, chi_tiet, pattern_20 = super_ai.predict(history)
            
            # Lưu dự đoán cho lần sau
            if du_doan in ['Tài', 'Xỉu']:
                last_prediction = du_doan
                last_conf = do_tin_cay
            else:
                last_prediction = None

            # Hiển thị thông tin
            print(f"    🔮 DỰ ĐOÁN PHIÊN TIẾP THEO: {du_doan}")
            print(f"       Độ tin cậy: {do_tin_cay}%")
            print(f"       Lý do: {ly_do}")
            print(f"       Pattern 20: {pattern_20}")
            
            if chi_tiet:
                # Hiển thị top 5 AI mạnh nhất
                top5 = sorted(chi_tiet, key=lambda x: x['vote'], reverse=True)[:5]
                print(f"       Top 5 AI: " + " | ".join([f"{t['name']}({t['confidence']:.0f}%)" for t in top5]))
            
            print("-" * 80)

            # Cập nhật JSON chuẩn
            latest_data = {
                "phiên": phien_id,
                "xúc_xắc_1": d1,
                "xúc_xắc_2": d2,
                "xúc_xắc_3": d3,
                "tổng": tong,
                "kết": ket_qua,
                "phiên_hiện_tại": phien_id,
                "dự_đoán": du_doan if du_doan in ['Tài', 'Xỉu'] else "Chờ",
                "pattern_20": pattern_20,
                "độ_tin_cậy": do_tin_cay,
                "lý_do": ly_do,
                "số_ai_hoạt_động": len(chi_tiet),
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
    return jsonify({
        "history": history_details[-50:],
        "pattern_full": pattern_all,
        "pattern_20_gần_nhất": pattern_all[-20:] if len(pattern_all) >= 20 else pattern_all,
        "số_phiên_đã_lưu": len(history),
        "độ_chính_xác": {
            "tổng": accuracy_tracker['total'],
            "đúng": accuracy_tracker['correct'],
            "tỷ_lệ": round(accuracy_tracker['correct'] / accuracy_tracker['total'] * 100, 2) if accuracy_tracker['total'] > 0 else 0
        }
    })


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    print("Server đang chạy trên port 10000...")
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)
