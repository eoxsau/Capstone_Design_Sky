import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter
import csv

# === 시스템 파라미터 ===
GROUP_COUNT = 40
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = 3.825
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.1
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

# === 복소수 위상 반복 함수
def generate_complex_phase(z: complex, iterations: int = 1):
    for _ in range(iterations):
        raw_val = abs(z) * GOLDEN_RATIO
        slice_val = int((raw_val * 1e10) % 1e10) % 839
        phi_zc = -np.pi * 25 * slice_val * (slice_val + 1) / 839
        rotator_zc = np.exp(1j * phi_zc)
        
        phi = 2 * np.pi * ((abs(z) * GOLDEN_RATIO) % 1)
        rotator = np.exp(1j * phi)
        z = (z * z + c) * rotator * rotator_zc

    angle = np.angle(z)
    norm = (angle + np.pi) / (2 * np.pi)
    return norm

def get_phase_from_aes(tod, trial, label, key, offset=0):
    pt = f"{tod}-{trial}-{label}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = AES.new(key, AES.MODE_ECB).encrypt(pt_hash)
    real = int.from_bytes(ct[offset:offset+4], 'big') / 1e9
    imag = int.from_bytes(ct[offset+4:offset+8], 'big') / 1e9
    z = complex(real, imag)
    return z

# === 도약 주파수 생성 함수 ===
def generate_group_frequencies(tod, trial, key):  # 주파수 위치 생성 함수
    group_freqs = []  # 그룹별 주파수 리스트

    for group_id in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        centers_freq = norm * TOTAL_BANDWIDTH_MHZ  # 중심 주파수 결정

        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(centers_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            z += 0.001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm = generate_complex_phase(z)
            centers_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break

        group_freqs.append(centers_freq)

    return group_freqs

# === 그룹 순서 결정 함수 ===
def generate_group_order(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=8)
        norm_list.append(generate_complex_phase(z))
    return list(np.argsort(norm_list))

# === 최종 테이블 ===
def final_table_gen(trials=10):
    key = b'0123456789abcdef0123456789abcdef'
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies(tod, trial, key)
        order = generate_group_order(tod, trial, key)
        final_freq = [table[rank] for rank in order]
        final_table.append(final_freq)

# === 메인 함수 ===
if __name__ == "__main__":
    a = time.perf_counter()
    final_table_gen(trials=100)
    b = time.perf_counter()

    print(b - a)