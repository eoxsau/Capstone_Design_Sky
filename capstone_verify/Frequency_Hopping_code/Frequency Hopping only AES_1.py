import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = 3.825
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.1  # 최소 이격 거리
AES_KEY = b'\x13\x6f\x95\x6e\xc6\x32\x20\x70\xc4\xb1\xd0\x73\x5b\x19\x29\x34\x0d\x9b\xaf\x32\x4a\xab\xe0\x46\x7e\xd4\xe4\x98\x17\x81\x09\x08'

# === AES 암호화 함수 (AES-256 ECB) ===
def aes_encrypt_block(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(data)

# === 주파수 위치 생성 함수 ===
def generate_group_frequencies(tod, trial, key):
    group_freqs = []
    used_freqs = []

    for group_id in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-{group_id}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)

        value = int.from_bytes(ct[:8], 'big')  # 앞 8바이트로 norm 생성
        norm = value / (2**64 - 1)
        base_freq = norm * TOTAL_BANDWIDTH_MHZ

        # 이격 거리 만족 검사
        attempts = 0
        while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in used_freqs):
            ct = aes_encrypt_block(ct, key)
            value = int.from_bytes(ct[:8], 'big')
            norm = value / (2**64 - 1)
            base_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:
                break

        used_freqs.append(base_freq)
        slots = [base_freq + i * SLOT_BANDWIDTH_MHZ for i in range(17)]
        group_freqs.append(slots)

    return group_freqs

# === 그룹 순서 생성 함수 ===
def generate_group_order(tod, trial, key):
    scores = []
    for rank in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-R{rank}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)
        value = int.from_bytes(ct[8:], 'big')  # 뒤 8바이트로 score 생성
        score = value / (2**64 - 1)
        scores.append((score, rank))
    scores.sort()
    return [rank for _, rank in scores]

# === 시각화 함수 ===
def visualize_all(trials=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14), sharex=True)
    tod = int(time.time()) & ((1 << 44) - 1)

    # 결과 캐싱
    all_freq_tables = [generate_group_frequencies(tod, trial, AES_KEY) for trial in range(trials)]
    all_orders = [generate_group_order(tod, trial, AES_KEY) for trial in range(trials)]

    # --- 주파수 위치 시각화 ---
    for group_id in range(GROUP_COUNT):
        freqs = [all_freq_tables[trial][group_id][0] for trial in range(trials)]
        ax1.plot(range(trials), freqs, alpha=0.6)

    ax1.set_title("Trial-wise Frequency Location Transitions")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_ylim(0, TOTAL_BANDWIDTH_MHZ)
    ax1.set_xlim(0, trials)
    ax1.grid(True)

    # --- 그룹 순서 시각화 ---
    group_history = [[] for _ in range(GROUP_COUNT)]
    for trial in range(trials):
        order = all_orders[trial]
        for idx, gid in enumerate(order):
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

# === 메인 함수 ===
if __name__ == "__main__":
    start = perf_counter()
    visualize_all(trials=10)
    end = perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")