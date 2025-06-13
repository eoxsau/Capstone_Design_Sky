import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.225  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = 3.825  # 그룹 하나당 대역폭 = 0.3825 MHz (0.225 * 17 = 3.825)
TOTAL_BANDWIDTH_MHZ = 200  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = 0.1  # 최소 이격 거리 = 슬롯 기준 5개 = 0.225 * 17 = 1.125 MHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

# === 복소수 위상 반복 함수
def generate_complex_phase(z: complex, iterations: int = 7):
    epsilon = 1e-7
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)
        z = np.exp(phi * 1j) + np.log(abs(z) + epsilon) + c

    angle = np.angle(z)
    raw = (angle + np.pi) / (2 * np.pi)

    norm = (np.sin(3 * np.pi * raw) + 1) / 2

    return norm

# === 한번의 시드로 복소수 생성 === (offset을 이용해서 그룹 순서를 반환할 수 있도록 바꿨음.)
def get_phase_from_aes(tod, trial, label, key, offset=0):
    pt = f"{tod}-{trial}-{label}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = AES.new(key, AES.MODE_ECB).encrypt(pt_hash)
    real = int.from_bytes(ct[offset:offset+4], 'big') / 1e9
    imag = int.from_bytes(ct[offset+4:offset+8], 'big') / 1e9
    return complex(real, imag)

# === 도약 주파수 생성 함수 ===
def generate_group_frequencies(tod, trial, key):  # 주파수 위치 생성 함수
    group_freqs = []  # 그룹별 주파수 리스트

    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=0)  # 복소수 시드 생성
        norm = generate_complex_phase(z)
        base_freq = norm * TOTAL_BANDWIDTH_MHZ  # 중심 주파수 결정

        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            z += 0.001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm = generate_complex_phase(z)
            base_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break

        group_freqs.append(base_freq)

    return group_freqs

# === 그룹 순서 결정 함수 ===
def generate_group_order(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=8)
        norm_list.append(generate_complex_phase(z))
    return list(np.argsort(norm_list))  # 크기 기준 정렬된 group_id 인덱스 반환

# === 최종 테이블 ===
def final_table_gen(trials, save_csv=True, filename="test_result/final-seq-V4.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies(tod, trial, key)
        order = generate_group_order(tod, trial, key)
        final_freq = [table[rank] for rank in order]
        final_table.append(final_freq)
    
    # CSV 파일로 저장
    if save_csv:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow(['Trial', 'Group', 'Frequency (MHz)'])
            
            # 데이터 작성
            for trial in range(trials):
                for group in range(GROUP_COUNT):
                    frequency = final_table[trial][group]
                    writer.writerow([trial, group, frequency])
    
    return final_table

# === 메인 함수 ===
if __name__ == "__main__":
    start = time.perf_counter()
    final_table_gen(10000)
    end = time.perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")