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
REST_SLOT = USABLE_SLOT - (GROUP_COUNT * SLOT_COUNT) - GROUP_COUNT

GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

class Hopping_ver7:

    def __init__(self):
        self.trial = 0
        self.key = b'\x13\x6f\x95\x6e\xc6\x32\x20\x70\xc4\xb1\xd0\x73\x5b\x19\x29\x34\x0d\x9b\xaf\x32\x4a\xab\xe0\x46\x7e\xd4\xe4\x98\x17\x81\x09\x08'

    # === 복소수 위상 반복 함수 ===
    def generate_complex_phase(self, z: complex, iterations: int = 1):
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

    # === CT로부터 위상 계산 ===
    def get_phase_from_aes(self, tod, label, offset=0):
        pt = f"{tod}-{self.trial}-{label}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = AES.new(self.key, AES.MODE_ECB).encrypt(pt_hash)
        real = int.from_bytes(ct[offset:offset + 4], 'big') / 1e9
        imag = int.from_bytes(ct[offset + 4:offset + 8], 'big') / 1e9
        z = complex(real, imag)
        return z
    
    # === 도약 주파수 생성 함수 [동적] ===
    def generate_group_frequencies(self, tod):
        z = self.get_phase_from_aes(tod, 0, offset=0)
        norm = self.generate_complex_phase(z)
        start_pos = int(norm * USABLE_SLOT)

        group_freqs = []
        for gid in range(1, GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=0)
            norm = self.generate_complex_phase(z)
            group_freqs.append(norm)
            
        error_sum = sum(group_freqs) / REST_SLOT
        group_freqs = [int(np.round(freq / error_sum)) for freq in group_freqs]

        group_freqs.insert(0, start_pos)
        for i in range(1, GROUP_COUNT):
            group_freqs[i] = group_freqs[i] + group_freqs[i - 1] + SLOT_COUNT
            group_freqs[i] %= USABLE_SLOT

        return group_freqs

    # === 그룹 순서 결정 함수 ===
    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=8)
            norm_list.append(self.generate_complex_phase(z))
        return list(np.argsort(norm_list))

    def init_csv(self, filename="test_result_modify/Ver7_modify.csv"):
        """CSV 파일 초기화 및 헤더 작성"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Trial', 'Group', 'Frequency'])

    # === 최종 테이블 ===
    def final_table_gen(self, save_csv=False, filename="test_result_modify/Ver7_modify.csv"):
        tod = int(time.time()) & ((1 << 44) - 1)

        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]

        # CSV 저장
        if save_csv:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for group in range(GROUP_COUNT):
                    frequency = final_freq[group]
                    writer.writerow([self.trial, group, frequency])

        self.trial += 1

        return table


# === 메인 함수 ===
if __name__ == "__main__":
    fhss = Hopping_ver7()
    
    # CSV 파일 초기화
    fhss.init_csv()

    start_time = time.perf_counter()
    for i in range(1000):
        final_table = fhss.final_table_gen(save_csv=True)
    end_time = time.perf_counter()

    print(f"실행 시간: {end_time - start_time:.6f}초")
    print("CSV 파일 저장 완료: Ver7_modify.csv")
