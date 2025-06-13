import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter
import csv
from datetime import datetime

# === 시스템 파라미터 ===
GROUP_COUNT = 40
SLOT_BANDWIDTH_MHZ = 225e6
GROUP_BANDWIDTH_MHZ = 3.825e6
TOTAL_BANDWIDTH_MHZ = 200e6
MIN_SPACING_MHZ = 100e3
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

BLOCK_BANDWIDTH = GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ
USABLE_BANDWIDTH = ((TOTAL_BANDWIDTH_MHZ - ((GROUP_COUNT - 1) * BLOCK_BANDWIDTH) - GROUP_BANDWIDTH_MHZ) / 4) * 3
USABLE_BANDWIDTH_2 = TOTAL_BANDWIDTH_MHZ - ((GROUP_COUNT - 1) * BLOCK_BANDWIDTH) - GROUP_BANDWIDTH_MHZ

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
            
        """
        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(centers_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            z += 0.01  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm = generate_complex_phase(z)
            centers_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break
        """
        group_freqs.append(centers_freq)

    return group_freqs

def generate_group_frequencies_modify(tod, trial, key):
    rest_band = TOTAL_BANDWIDTH_MHZ - (GROUP_BANDWIDTH_MHZ * GROUP_COUNT)
    
    z = get_phase_from_aes(tod, trial, 40, key, offset=0)
    norm = generate_complex_phase(z)
    base_freq = norm * TOTAL_BANDWIDTH_MHZ

    diff = int(rest_band / GROUP_COUNT)

    group_freqs = [base_freq]
    for _ in range(1, GROUP_COUNT):
        base_freq += GROUP_BANDWIDTH_MHZ + diff
        
        base_freq %= TOTAL_BANDWIDTH_MHZ

        group_freqs.append(base_freq)

    return group_freqs

# === 도약 주파수 생성 함수 ===
def generate_group_frequencies_modify_2(tod, trial, key):  # 주파수 위치 생성 함수
    z = get_phase_from_aes(tod, trial, 0, key, offset=0)
    norm = generate_complex_phase(z)
    start_pos = norm * GROUP_BANDWIDTH_MHZ

    #print(USABLE_BANDWIDTH)

    # 1. 랜덤 주파수 생성 (복소 위상 기반)
    initial_freqs = []  # 초기 랜덤 중심 주파수 리스트
    for group_id in range(1, GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        initial_freqs.append(norm)

    # 2. 정렬 (정규화된 0~1 값 기준)
    initial_freqs.sort()

    # 4. 정규화된 값들을 usable_band 안으로 매핑
    scaled_freqs = [r * USABLE_BANDWIDTH for r in initial_freqs]

    # 5. 이격 거리만큼 보정하여 최종 주파수 리스트 생성
    adjusted_freqs = [(start_pos + f + (i + 1) * BLOCK_BANDWIDTH) % TOTAL_BANDWIDTH_MHZ for i, f in enumerate(scaled_freqs)]
    adjusted_freqs.insert(0, start_pos)

    return adjusted_freqs

# === 그룹 순서 결정 함수 ===
def generate_group_order(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=8)
        norm_list.append(generate_complex_phase(z))
    return list(np.argsort(norm_list))

def cal_distance(table:list):
    # --- 정렬 전 table index 저장 ---
    index_table = np.argsort(np.argsort(table))
    
    # --- table 정렬 ---
    sorted_table = sorted(table)
    
    # --- 이격 저장 ---
    distance = [sorted_table[0]]
    if distance[0] >= GROUP_BANDWIDTH_MHZ:
        distance[0] -= GROUP_BANDWIDTH_MHZ
    
    # --- 앞의 이격을 구하면 뒤 이격이 달라지므로 이격을 계산하면서 바로바로 조건을 체크해야 함 ---
    for i in range(1, GROUP_COUNT):
        dis = (sorted_table[i] - sorted_table[i - 1]) - (distance[i-1] - sorted_table[i - 1])
        
        if dis < MIN_SPACING_MHZ + GROUP_BANDWIDTH_MHZ:
            #a = MIN_SPACING_MHZ + GROUP_BANDWIDTH_MHZ - dis
            #dis += (1.01*a)
            dis = MIN_SPACING_MHZ + GROUP_BANDWIDTH_MHZ
        
        while dis >= (MIN_SPACING_MHZ + 1.4 * GROUP_BANDWIDTH_MHZ):
            dis -= 0.4 * GROUP_BANDWIDTH_MHZ

        distance.append(dis + distance[i - 1])
 
    # --- table 복구 ---
    recover_table = [distance[i] for i in index_table]
    
    return recover_table

# === 최종 테이블 ===
def final_table_gen(trials, save_csv=True, filename="test_result/final-seq-V5.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies_modify_2(tod, trial, key)
        #a = cal_distance(table)
        
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
                    frequency = round(final_table[trial][group], 3)
                    writer.writerow([trial, group, frequency])
    
    return final_table


# === 메인 함수 ===
if __name__ == "__main__":

    start = time.perf_counter()
    final_table = final_table_gen(1000)  # 빠른 분석을 위해 1000으로 줄임
    end = time.perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")
