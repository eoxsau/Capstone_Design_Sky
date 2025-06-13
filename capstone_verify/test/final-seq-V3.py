import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 225e3  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = 3.825e6  # 그룹 하나당 대역폭 = 0.3825 MHz (0.225 * 17 = 3.825)
TOTAL_BANDWIDTH_MHZ = 200e6  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = 100e3  # 최소 이격 거리 = 슬롯 기준 5개 = 0.225 * 17 = 1.125 MHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

# === 복소수 위상 기반 정규화 ===
def generate_complex_phase(z: complex, iterations: int = 7):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)  # 위상값 계산 (phi를 이렇게 정의한 이유는 더욱 복잡하게 만들기 위해 z의 실수값과 황금비를 곱하고 그걸 0부터 1까지의 값으로 정규화함.)
        rotator = np.exp(1j * phi)  # 회전 연산자 (오일러 함수)
        z = (z*z + c) * rotator  # 회전 및 복소수 제곱 연산
    angle = np.angle(z)  # 복소수의 위상 (역연산으로 정확한 복소수 좌표를 얻기 힘듦. 해당 좌표와 원점이 이루는 선상의 모든 점의 좌표가 경우의 수가 되기 때문)
    norm = (angle + np.pi) / (2 * np.pi)
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

        """
        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            z += 0.001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm = generate_complex_phase(z)
            base_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break
        """

        group_freqs.append(base_freq)

    return group_freqs

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
 
    print(distance[39])
    # --- table 복구 ---
    recover_table = [distance[i] for i in index_table]
    
    return recover_table

def generate_group_frequencies_modify(tod, trial, key):
    rest_band = TOTAL_BANDWIDTH_MHZ - (GROUP_BANDWIDTH_MHZ * GROUP_COUNT)
    
    z = get_phase_from_aes(tod, trial, 0, key, offset=0)
    norm = generate_complex_phase(z)
    base_freq = norm * TOTAL_BANDWIDTH_MHZ

    group_freqs = [base_freq]
    for _ in range(1, GROUP_COUNT):
        base_freq += GROUP_BANDWIDTH_MHZ + int(rest_band / GROUP_COUNT)
        
        base_freq %= TOTAL_BANDWIDTH_MHZ

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
def final_table_gen(trials, save_csv=True, filename="test_result/final-seq-V3.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        #table = generate_group_frequencies(tod, trial, key)
        #a = cal_distance(table)
        
        table = generate_group_frequencies_modify(tod, trial, key)
        
        order = generate_group_order(tod, trial, key)
        final_freq = [table[rank] for rank in order]
        final_table.append(final_freq)
    
    print(max(final_table[0]))
    
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