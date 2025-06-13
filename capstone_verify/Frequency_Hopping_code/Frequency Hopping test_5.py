import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter
import csv

# === 시스템 파라미터 ===
GROUP_COUNT = 45
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = 3.825 # SLOT 크기 x 17
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.1
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
AES_KEY = b'0123456789abcdef0123456789abcdef'

# === 복소수 위상 반복 함수
def generate_complex_phase(z: complex, iterations: int = 7):
    N = 839  # ZC 시퀀스 길이 (소수이어야 함.)
    u = 25   # ZC 루트

    GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

    # Zadoff–Chu
    index = int((abs(z) % 1.0) * N)
    phi_zc = -np.pi * u * index * (index + 1) / N
    rotator_zc = np.exp(1j * phi_zc)

    # 복소 반복 수식
    for _ in range(iterations):
        phi = 2 * np.pi * ((abs(z) * GOLDEN_RATIO) % 1)
        rotator = np.exp(1j * phi)
        z = (z * z + c) * rotator * rotator_zc  # ZC를 보조 회전으로 사용.

    # 위상 추출 및 정규화
    angle = np.angle(z)
    norm = (angle + np.pi) / (2 * np.pi)

    # 텐트 맵 적용 (비선형 분산)
    p = 0.5
    for _ in range(4):
        norm = norm / p if norm < p else (1 - norm) / (1 - p)
        norm %= 1.0
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

            real1 = int.from_bytes(ct[:4], 'big') / 1e9
            imag1 = int.from_bytes(ct[4:8], 'big') / 1e9
            z_freq = complex(real1, imag1)
            freq_norm, _ = generate_complex_phase(z_freq)
            base_freq = freq_norm * TOTAL_BANDWIDTH_MHZ

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

    # === 최종 도약 테이블 저장 ===
    with open("final_hopping_table.txt", "w") as txt_file, \
         open("final_hopping_table.csv", "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["Trial", "Rank", "Group_ID", "Slot_8 (MHz)"])

        for trial in range(trials):
            order = group_orders[trial]
            group_freqs = freq_tables[trial]

            txt_file.write(f"=== Trial {trial} ===\n")

            for rank, gid in enumerate(order):
                slot8 = round(group_freqs[gid][8], 6)  # slot 8만 사용
                txt_file.write(f"Rank {rank} (Group {gid}): {slot8:.6f} MHz\n")
                writer.writerow([trial, rank, gid, slot8])

            txt_file.write("\n")

# === 메인 실행 ===
if __name__ == "__main__":
    start = perf_counter()
    visualize_all(trials=10)
    end = perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")