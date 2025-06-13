import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter
import secrets
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 225e3  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = 3.825e6  # 그룹 하나당 대역폭 = 0.3825 MHz (0.0225 * 17 = 0.3825)
TOTAL_BANDWIDTH_MHZ = 200e6  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = 100e3
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
# 16진수 문자열을 바이너리로 변환하여 32바이트 AES-256 키 생성
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

# === AES 암호화 함수 (AES-256 ECB) ===
def aes_encrypt_block(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(data)

# === 주파수 위치 생성 함수 ===
def generate_group_frequencies_AES(tod, trial, key):
    group_freqs = []

    for group_id in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-{group_id}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)

        value = int.from_bytes(ct[:8], 'big')  # 앞 8바이트로 norm 생성
        norm = value / (2**64 - 1)
        base_freq = norm * TOTAL_BANDWIDTH_MHZ

        # 이격 거리 만족 검사
        """
        attempts = 0
        while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in used_freqs):
            #ct = aes_encrypt_block(ct, key)
            #value = int.from_bytes(ct[:8], 'big')
            #norm = value / (2**64 - 1)
            #base_freq = norm * TOTAL_BANDWIDTH_MHZ
            
            base_freq += 0.05
            if base_freq > 195:
                base_freq = 0
            
            #attempts += 1
            #if attempts > 1000:
            #    base_freq = 0
            #    break
        """

        group_freqs.append(base_freq)

    return group_freqs

def generate_group_frequencies_modify(tod, trial, key):
    rest_band = TOTAL_BANDWIDTH_MHZ - (GROUP_BANDWIDTH_MHZ * GROUP_COUNT)
    
    pt = f"{tod}-{trial}-{0}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = aes_encrypt_block(pt_hash, key)

    value = int.from_bytes(ct[:8], 'big')  # 앞 8바이트로 norm 생성
    norm = value / (2**64 - 1)
    base_freq = norm * TOTAL_BANDWIDTH_MHZ

    group_freqs = [base_freq]
    for _ in range(1, GROUP_COUNT):
        base_freq += GROUP_BANDWIDTH_MHZ + int(rest_band / GROUP_COUNT)
        
        base_freq %= TOTAL_BANDWIDTH_MHZ

        group_freqs.append(base_freq)

    return group_freqs

# === 그룹 순서 생성 함수 ===
def generate_group_order_AES(tod, trial, key):
    scores = []
    for rank in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-{rank}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)
        value = int.from_bytes(ct[8:], 'big')  # 뒤 8바이트로 score 생성
        score = value / (2**64 - 1)
        scores.append((score, rank))
    scores.sort()
    return [rank for _, rank in scores]
                    
# === 최종 테이블 ===
def final_table_gen(trials, save_csv=True, filename="test_result/final-seq-AES.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies_modify(tod, trial, key)
        order = generate_group_order_AES(tod, trial, key)
        
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
    start = perf_counter()
    a = final_table_gen(10000)
    end = perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")
