import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter

# === 시스템 파라미터 ===
GROUP_COUNT = 45
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = SLOT_BANDWIDTH_MHZ * 17
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.06
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
AES_KEY = b'0123456789abcdef0123456789abcdef'

# === 복소수 위상 반복 함수
def generate_complex_phase(z: complex, iterations: int = 7):
    epsilon = 1e-7
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)
        z = np.exp(phi * 1j) + np.log(abs(z) + epsilon) + c

    angle = np.angle(z)
    raw = (angle + np.pi) / (2 * np.pi)

    # 중앙 집중 완화용 flattening 함수
    norm = (np.sin(3 * np.pi * raw) + 1) / 2

    return norm, angle

# === 시각화 및 전체 도약 로직
def visualize_all(trials=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14), sharex=True)
    tod = int(time.time()) & ((1 << 44) - 1)

    freq_tables = []
    group_orders = []

    for trial in range(trials):
        freqs = []
        scores = []
        used_freqs = []

        for group_id in range(GROUP_COUNT):
            pt = f"{tod}-{trial}-{group_id}".encode()
            pt_hash = sha256(pt).digest()[:16]
            ct = AES.new(AES_KEY, AES.MODE_ECB).encrypt(pt_hash)

            # 앞 8바이트 → 주파수 위치
            real1 = int.from_bytes(ct[:4], 'big') / 1e9
            imag1 = int.from_bytes(ct[4:8], 'big') / 1e9
            z_freq = complex(real1, imag1)
            freq_norm, _ = generate_complex_phase(z_freq)
            base_freq = freq_norm * TOTAL_BANDWIDTH_MHZ

            # 홀수 group_id일 경우 주파수 반전
            if (group_id % 2 == 1) and (group_id < 100):
                base_freq = TOTAL_BANDWIDTH_MHZ - (base_freq / 2)
            elif (group_id % 2 == 1) and (group_id > 100):
                base_freq = TOTAL_BANDWIDTH_MHZ - 100 - (base_freq / 2)

            # 최소 이격 거리 보장
            attempts = 0
            while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in used_freqs):
                freq_norm += 0.000001
                base_freq = (freq_norm % 1.0) * TOTAL_BANDWIDTH_MHZ
                if (group_id % 2 == 1) and (group_id < 100):
                    base_freq = TOTAL_BANDWIDTH_MHZ - (base_freq / 2)
                elif (group_id % 2 == 1) and (group_id > 100):
                    base_freq = TOTAL_BANDWIDTH_MHZ - 100 - (base_freq / 2)
                attempts += 1
                if attempts > 1000:
                    break

            used_freqs.append(base_freq)
            slots = [base_freq + i * SLOT_BANDWIDTH_MHZ for i in range(17)]
            freqs.append(slots)

            # 뒤 8바이트 → 그룹 순서 점수
            real2 = int.from_bytes(ct[8:12], 'big') / 1e9
            imag2 = int.from_bytes(ct[12:16], 'big') / 1e9
            z_rank = complex(real2, imag2)
            score, _ = generate_complex_phase(z_rank)
            scores.append((score, group_id))

        scores.sort()
        group_order = [gid for _, gid in scores]
        freq_tables.append(freqs)
        group_orders.append(group_order)

    # === 주파수 위치 시각화 ===
    for group_id in range(GROUP_COUNT):
        f = [freq_tables[trial][group_id][0] for trial in range(trials)]
        ax1.plot(range(trials), f, alpha=0.6)
    ax1.set_title("Trial-wise Frequency Location Transitions")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_ylim(0, TOTAL_BANDWIDTH_MHZ)
    ax1.set_xlim(0, trials)
    ax1.grid(True)

    # === 그룹 순서 시각화 ===
    group_history = [[] for _ in range(GROUP_COUNT)]
    for trial in range(trials):
        for idx, gid in enumerate(group_orders[trial]):
            group_history[gid].append(idx)
    for gid in range(GROUP_COUNT):
        ax2.plot(range(trials), group_history[gid], alpha=0.5)
    ax2.set_title("Trial-wise Group Order Transitions")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Rank")
    ax2.set_ylim(0, GROUP_COUNT)
    ax2.set_xlim(0, trials)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# === 메인 실행 ===
if __name__ == "__main__":
    start = perf_counter()
    visualize_all(trials=10)
    end = perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")