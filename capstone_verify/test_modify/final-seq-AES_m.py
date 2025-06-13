import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40
SLOT_COUNT = 17
TOTAL_BANDWIDTH = 200e6
SLOT_BANDWIDTH = 225e3

TOTAL_SLOT_INDEX = int(TOTAL_BANDWIDTH / SLOT_BANDWIDTH)
USABLE_SLOT = TOTAL_SLOT_INDEX - SLOT_COUNT
REST_SLOT = USABLE_SLOT - (GROUP_COUNT * SLOT_COUNT)

GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

# === AES 암호화 함수 (AES-256 ECB) ===
def aes_encrypt_block(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(data)

# === 도약 주파수 생성 함수 [재배치] ===
def generate_group_frequencies_1(tod, trial, key):

    # 1. 랜덤 주파수 생성 (복소 위상 기반)
    group_freqs = []
    for gid in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-{gid}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)

        value = int.from_bytes(ct[:8], 'big')
        norm = value / (2**64 - 1)
        centers_freq = int(norm * USABLE_SLOT)

        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(centers_freq - f) < (SLOT_COUNT + 1) for f in group_freqs):
            ct = aes_encrypt_block(ct, key)
            value = int.from_bytes(ct[:8], 'big')
            norm = value / (2**64 - 1)
            centers_freq = int(norm * USABLE_SLOT)
            
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break

        group_freqs.append(centers_freq)

    return group_freqs

# === 주파수 위치 생성 함수 [정적] ===
def generate_group_frequencies_2(tod, trial, key):
    pt = f"{tod}-{trial}-{0}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = aes_encrypt_block(pt_hash, key)

    value = int.from_bytes(ct[:8], 'big')
    norm = value / (2**64 - 1)
    base_freq = int(norm * USABLE_SLOT)

    group_freqs = [base_freq]
    for _ in range(1, GROUP_COUNT):
        base_freq += GROUP_COUNT + int(REST_SLOT / GROUP_COUNT)
        base_freq %= USABLE_SLOT

        group_freqs.append(base_freq)

    return group_freqs

# === 주파수 위치 생성 함수 [동적] ===
def generate_group_frequencies_3(tod, trial, key):
    pt = f"{tod}-{trial}-{0}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = aes_encrypt_block(pt_hash, key)
    value = int.from_bytes(ct[:8], 'big')  # 앞 8바이트로 norm 생성
    norm = value / (2 ** 64 - 1)
    start_pos = int(norm * USABLE_SLOT)

    # 1. 랜덤 주파수 생성 (복소 위상 기반)
    initial_freqs = []  # 초기 랜덤 중심 주파수 리스트
    for gid in range(1, GROUP_COUNT):
        pt = f"{tod}-{trial}-{gid}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)
        value = int.from_bytes(ct[:8], 'big')  # 앞 8바이트로 norm 생성
        norm = value / (2 ** 64 - 1)
        initial_freqs.append(norm)

    # 2. 정렬 (정규화된 0~1 값 기준)
    initial_freqs.sort()

    # 4. 정규화된 값들을 usable_band 안으로 매핑
    scaled_freqs = [int(r * REST_SLOT) for r in initial_freqs]

    # 5. 이격 거리만큼 보정하여 최종 주파수 리스트 생성
    adjusted_freqs = [(start_pos + f * (i + 1)) % USABLE_SLOT for i, f in enumerate(scaled_freqs)]
    adjusted_freqs.insert(0, start_pos)

    return adjusted_freqs

# === 그룹 순서 생성 함수 ===
def generate_group_order_AES(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        pt = f"{tod}-{trial}-{gid}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = aes_encrypt_block(pt_hash, key)
        value = int.from_bytes(ct[8:], 'big')  # 뒤 8바이트로 score 생성
        norm = value / (2**64 - 1)
        norm_list.append(norm)
        
    return list(np.argsort(norm_list))
                    
# === 최종 테이블 ===
def final_table_gen(trials, save_csv=True, filename="test_result_modify/VerAES_modify.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)

    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies_1(tod, trial, key)
        order = generate_group_order_AES(tod, trial, key)
        
        final_freq = [table[rank] for rank in order]
        final_table.append(final_freq)
    
    # CSV 파일로 저장
    if save_csv:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Trial', 'Group', 'Frequency'])
            
            # 데이터 작성
            for trial in range(trials):
                for group in range(GROUP_COUNT):
                    frequency = final_table[trial][group]
                    writer.writerow([trial, group, frequency])
    
    return final_table
                    
# === 메인 함수 ===
if __name__ == "__main__":
    start = time.perf_counter()
    a = final_table_gen(1000)
    end = time.perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")
