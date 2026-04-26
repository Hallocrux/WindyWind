#!/usr/bin/env python3
"""
风力发电机风速预测工具 v2.0
基于振动特征 + k-NN 工况匹配

更新日志 v2.0:
- 新增3组校正数据(实际风速5.8, 4.8, 5.5 m/s)
- 修正风速-RPM线性关系: wind = 0.0222*rpm + 0.383
- 增加FFT可信度检测 (RPM区间校验)
- k-NN特征匹配备用方案
"""

import os
import sys
import csv
import re
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# ============================================================
# 已知工况数据库 (FFT检测转速, RMS, Peak, Crest, 实际风速)
# ============================================================
KNOWNCASES = [
    # (fft_rpm, rms, peak, crest, wind)
    (146, 0.9206, 19.2254, 20.88, 2.12),   # Case1  82rpm
    (155, 0.9698,  6.7500,  6.96, 3.11),   # Case3 158rpm
    (126, 1.5323, 19.7559, 12.89, 4.50),   # Case4 172rpm
    (132, 1.6482, 19.8803, 12.06, 3.90),   # Case5 166rpm
    (129, 0.4697,  2.2508,  4.79, 4.60),   # Case6 195rpm
    (144, 0.5140,  5.1227,  9.97, 4.15),   # Case7 155rpm
    (132, 0.7433, 19.7559,  4.93, 4.90),   # Case8 210rpm
    (161, 1.0773, 13.5830, 13.58, 4.30),   # Case9 159rpm
    (129, 0.4575,  2.4927,  5.45, 4.60),   # Case10 180rpm
    (132, 0.5703,  4.0640,  7.13, 4.70),   # Case11 190rpm
    (164, 1.0000, 10.0000, 10.00, 5.30),   # Case12 220rpm
    (146, 0.5000,  3.0000,  6.00, 5.38),   # Case14 230rpm
    # v2.0 新增校正数据
    (132, 2.0400, 15.1600,  7.42, 5.80),   # 补充1 实际5.8m/s
    (129, 0.4580,  2.0177,  4.41, 4.80),   # 补充2 实际4.8m/s
    (220, 0.8235,  4.4862,  5.45, 5.50),   # 补充3 实际5.5m/s
    # v3.0 工况13/16/17 实际值校正
    (226, 1.0042,  6.2815,  6.26, 6.50),   # Case13 实际6.5m/s
    (252, 1.1067,  8.3045,  7.50, 6.90),   # Case16 实际6.9m/s
    (260, 1.0476,  5.1817,  4.95, 8.10),   # Case17 实际8.1m/s
    # v4.0 2026-04-09 校正数据 (8个新验证工况)
    (215, 0.6660,  5.3366,  8.01, 6.20),   # 24-4补充工况 实际6.2m/s
    (165, 0.6296,  5.5768,  8.86, 5.30),   # 24-3补充工况 实际5.3m/s
    (109, 0.1303,  0.9902,  7.60, 3.90),   # 24-3补充工况2 实际3.9m/s
    (132, 0.7491,  4.3406,  5.79, 4.60),   # 23-2补充工况 实际4.6m/s
    (239, 0.4286,  2.5518,  5.95, 7.50),   # 23-2补充工况2 实际7.5m/s
    (233, 0.6743,  4.3390,  6.43, 6.80),   # 23-2补充工况3 实际6.8m/s
    (131, 1.7732, 19.4650, 10.98, 5.80),   # 23-2补充工况5 实际5.8m/s
    (122, 0.4228,  3.3277,  7.87, 6.20),   # 23-2补充工况6 实际6.2m/s
]

# 线性校正: wind = a * rpm + b
# 基于原始12工况 + 工况13/16/17(实际6.5/6.9/8.1m/s) 拟合
WIND_A = 0.0302
WIND_B = -0.8849

# 2026-04-09 校正后的新线性参数 (基于>140rpm的校正数据)
WIND_A_V2 = 0.02630
WIND_B_V2 = 0.8480

def fft_rpm_to_wind(rpm):
    """FFT转速 → 风速 (线性模型)"""
    return WIND_A * rpm + WIND_B

def extract_features(filepath):
    """从CSV提取振动特征 (增强FFT 1X检测)"""
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)

    # 找加速度列
    acc_cols = {}
    for i, h in enumerate(header):
        h_clean = h.strip()
        if '.AccX' in h or 'AccX' in h:
            acc_cols.setdefault('AccX', []).append(i)
        elif '.AccY' in h or 'AccY' in h:
            acc_cols.setdefault('AccY', []).append(i)
        elif '.AccZ' in h or 'AccZ' in h:
            acc_cols.setdefault('AccZ', []).append(i)

    # 优先用AccX (WSMS00003)
    target_cols = acc_cols.get('AccX', [])
    if not target_cols:
        for cols in acc_cols.values():
            target_cols.extend(cols)

    if not target_cols:
        return None

    col = target_cols[0]
    vals = []
    for row in rows:
        if len(row) > col:
            try:
                v = float(row[col])
                vals.append(v)
            except:
                pass

    if len(vals) < 50:
        return None

    arr = np.array(vals)
    rms = float(np.sqrt(np.mean(arr**2)))
    peak = float(np.max(np.abs(arr)))
    crest = peak / rms if rms > 0 else 0

    # FFT
    n = len(arr)
    fs = 50.0
    fft_vals = np.fft.fft(arr)
    freqs = np.fft.fftfreq(n, 1.0/fs)

    # 低频范围 (0.8~6Hz) - 对应48~360rpm，实际工况在120~250rpm → 2~4.2Hz
    lf_mask = (freqs >= 0.8) & (freqs <= 6.0)
    lf_freqs = freqs[lf_mask]
    lf_mags = np.abs(fft_vals[lf_mask])

    # 宽频率范围 (0.8~25Hz) - 用于整体分析
    pos_mask = (freqs >= 0.8) & (freqs < 25)
    pos_freqs = freqs[pos_mask]
    pos_mags = np.abs(fft_vals[pos_mask])

    # 方法1: 在低频带找1X (候选转速频率)
    # 取低频带top 3 peaks
    top_lf_idx = np.argsort(lf_mags)[::-1][:5]
    candidates = []
    for idx in top_lf_idx:
        f = lf_freqs[idx]
        m = lf_mags[idx]
        # 谐波评分: 1X + 2X + 3X 的能量
        harm_score = float(m)  # 1X能量
        for h in [2, 3, 4]:
            hz = h * f
            if hz < 25:
                bin_idx = int(round(hz / (fs / n)))
                if 0 <= bin_idx < len(pos_mags):
                    harm_score += float(pos_mags[bin_idx])
        candidates.append((f, m, harm_score))

    # 谐波评分最高的作为1X
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_f, best_mag, harm_score = candidates[0]
    rpm_est = best_f * 60.0

    # 谐波搜索
    harmonics = {}
    for h in range(1, 6):
        target = h * best_f
        if target < 25:
            idx = int(round(target / (fs / n)))
            if 0 <= idx < len(pos_mags):
                harmonics[f'{h}X'] = float(pos_mags[idx])

    # 全频段峰值
    overall_peak_idx = np.argmax(pos_mags)
    overall_peak_freq = pos_freqs[overall_peak_idx]

    # 可信度评估
    rpm_in_range = 110 <= rpm_est <= 260
    crest_normal = crest < 15
    # 检查1X是否在低频带内且是主要峰值
    low_freq_dominant = rpm_est < 6.0  # 1X频率 < 6Hz
    harmonics_good = harmonics.get('2X', 0) > harmonics.get('1X', 0) * 0.1  # 2X至少有1X的10%
    confidence = 'HIGH' if (rpm_in_range and crest_normal and low_freq_dominant) else ('MED' if rpm_in_range else 'LOW')

    return {
        'rms': rms,
        'peak': peak,
        'crest': crest,
        'fft_rpm': rpm_est,
        'fft_freq': best_f,
        'fft_mag': best_mag,
        'harmonics': harmonics,
        'confidence': confidence,
        'rpm_in_range': rpm_in_range,
        'n_samples': len(vals),
    }


def knn_predict(features, k=3):
    """k-NN 特征匹配预测风速"""
    known = np.array([[c[0], c[1], c[2], c[3]] for c in KNOWNCASES])  # rpm, rms, peak, crest
    winds = np.array([c[4] for c in KNOWNCASES])

    # 标准化
    mean = known.mean(axis=0)
    std = known.std(axis=0) + 1e-9
    scaled = (known - mean) / std

    feat = np.array([features['fft_rpm'], features['rms'], features['peak'], features['crest']])
    scaled_feat = (feat - mean) / std

    # 距离加权
    dists = np.sqrt(((scaled - scaled_feat)**2).sum(axis=1))
    if k > len(dists):
        k = len(dists)
    k_nearest_idx = np.argsort(dists)[:k]
    k_dists = dists[k_nearest_idx]
    k_winds = winds[k_nearest_idx]

    # 距离加权平均
    weights = 1.0 / (k_dists + 0.01)
    weights /= weights.sum()
    pred_wind = np.dot(weights, k_winds)

    return float(pred_wind), [(float(KNOWNCASES[i][4]), float(k_dists[j])) for j, i in enumerate(k_nearest_idx)]


def predict(filepath):
    """主预测函数"""
    features = extract_features(filepath)
    if features is None:
        return {'error': '无法提取振动特征'}

    fft_wind = fft_rpm_to_wind(features['fft_rpm'])
    knn_wind, nn_info = knn_predict(features, k=3)

    # 集成: FFT+ k-NN 平均
    # FFT可信度高时偏重FFT，低时偏重k-NN
    if features['confidence'] == 'HIGH':
        pred_wind = 0.6 * fft_wind + 0.4 * knn_wind
    elif features['confidence'] == 'MED':
        pred_wind = 0.5 * fft_wind + 0.5 * knn_wind
    else:
        # LOW置信度: 主要靠k-NN
        pred_wind = 0.3 * fft_wind + 0.7 * knn_wind

    return {
        'fft_rpm': features['fft_rpm'],
        'fft_wind': fft_wind,
        'knn_wind': knn_wind,
        'pred_wind': pred_wind,
        'rms': features['rms'],
        'peak': features['peak'],
        'crest': features['crest'],
        'fft_freq': features['fft_freq'],
        'confidence': features['confidence'],
        'nn_info': nn_info,
    }


def print_result(r, filepath):
    basename = os.path.basename(filepath)
    conf_colors = {'HIGH': '✅', 'MED': '⚠️', 'LOW': '❌'}
    conf_label = {'HIGH': '高 (±0.5m/s)', 'MED': '中 (±1m/s)', 'LOW': '低 (±1.5m/s+)'}
    icon = conf_colors.get(r['confidence'], '❓')

    print(f'{icon} 文件: {basename}')
    print(f'   振动特征: RMS={r["rms"]:.4f}g  Peak={r["peak"]:.4f}g  Crest={r["crest"]:.2f}')
    print(f'   FFT转频: {r["fft_freq"]:.3f}Hz → {r["fft_rpm"]:.0f} rpm')
    print(f'   ── Prediction ──')
    print(f'   FFT模型:   {r["fft_wind"]:.2f} m/s')
    print(f'   k-NN模型:  {r["knn_wind"]:.2f} m/s')
    print(f'   ★ 集成预测: {r["pred_wind"]:.2f} m/s  ({conf_label.get(r["confidence"], "N/A")})')
    nn = r.get('nn_info', [])
    if nn:
        print(f'   k-NN匹配: {nn[0][0]:.2f}m/s(距离{nn[0][1]:.2f}) | {nn[1][0]:.2f}m/s({nn[1][1]:.2f}) | {nn[2][0]:.2f}m/s({nn[2][1]:.2f})')
    print()


def main():
    if len(sys.argv) < 2:
        print('用法: python3 predict.py <csv文件> [csv文件2 ...]')
        sys.exit(1)

    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f'文件不存在: {filepath}')
            continue
        r = predict(filepath)
        if 'error' in r:
            print(f'错误: {r["error"]}')
        else:
            print_result(r, filepath)


if __name__ == '__main__':
    main()
