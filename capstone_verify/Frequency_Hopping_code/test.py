import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH = 225e3  # 슬롯 크기 (0.225MHz = 225KHz)
GROUP_BANDWIDTH = 3.825e6  # 그룹 하나당 대역폭 (3.825MHz = 3825KHz)
MIN_SPACING = 100e3  # 최소 이격 거리 = 100KHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용

TOTAL_BANDWIDTH = 200e6  # 전체 사용 가능한 대역폭
BLOCK_BANDWIDTH = GROUP_BANDWIDTH + MIN_SPACING  # 그룹 대역폭 + 최소 이격 거리 = 이격 거리까지 포함하여 하나의 그룹으로 해석
REST_BANDWIDTH = TOTAL_BANDWIDTH - BLOCK_BANDWIDTH * GROUP_COUNT  # 전체 대역폭에서 실제 사용되는 전체 대역폭을 제외한 나머지 대역폭 (우리의 예제는 200 - 3.925 * 40 = 43MHz 남음)


class Hopping_ver7:

    def __init__(self):
        self.c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
        self.trial = 0
        self.key = b'\x13\x6f\x95\x6e\xc6\x32\x20\x70\xc4\xb1\xd0\x73\x5b\x19\x29\x34\x0d\x9b\xaf\x32\x4a\xab\xe0\x46\x7e\xd4\xe4\x98\x17\x81\x09\x08'

    # === 복소수 위상 반복 함수
    def generate_complex_phase(self, z: complex, iterations: int = 1):
        for _ in range(iterations):
            raw_val = abs(z) * GOLDEN_RATIO
            slice_val = int((raw_val * 1e10) % 1e10) % 839
            phi_zc = -np.pi * 25 * slice_val * (slice_val + 1) / 839
            rotator_zc = np.exp(1j * phi_zc)
            phi = 2 * np.pi * ((abs(z) * GOLDEN_RATIO) % 1)
            rotator = np.exp(1j * phi)
            z = (z * z + self.c) * rotator * rotator_zc

        angle = np.angle(z)
        norm = (angle + np.pi) / (2 * np.pi)
        return norm

    def get_phase_from_aes(self, tod, label, offset=0):
        pt = f"{tod}-{self.trial}-{label}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = AES.new(self.key, AES.MODE_ECB).encrypt(pt_hash)
        real = int.from_bytes(ct[offset:offset + 4], 'big') / 1e9
        imag = int.from_bytes(ct[offset + 4:offset + 8], 'big') / 1e9
        z = complex(real, imag)
        return z

    # === 도약 주파수 생성 함수 ===
    def generate_group_frequencies(self, tod):
        group_freqs = []

        for gid in range(GROUP_COUNT + 1):
            z = self.get_phase_from_aes(tod, gid, offset=0)
            norm = self.generate_complex_phase(z)
            group_freqs.append(norm)

        # --- 0~1 사이의 정규화 값을 모두 더하면 1이 되도록 계산 ---
        error_sum = sum(group_freqs)
        group_freqs = [freq / error_sum for freq in group_freqs]

        group_freqs = [freq * REST_BANDWIDTH for freq in group_freqs]

        # --- 이전 이격으로부터 블럭 대역폭에 다음 이격을 더하면 다음 주파수 위치 ---
        #for k in range(1, GROUP_COUNT):
        #    group_freqs[k] = group_freqs[k] + group_freqs[k - 1] + BLOCK_BANDWIDTH

        # --- 마지막 이격 정보는 필요없음 ---
        group_freqs.pop()
        
        
        
        return group_freqs

    # === 그룹 순서 결정 함수 ===
    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=8)
            norm_list.append(self.generate_complex_phase(z))
        return list(np.argsort(norm_list))

    def init_csv(self, filename="final-seq-V7.csv"):
        """CSV 파일 초기화 및 헤더 작성"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Trial', 'Group', 'Frequency (MHz)'])

    # === 최종 테이블 ===
    def final_table_gen(self, save_csv=False, filename="final-seq-V7.csv"):
        tod = int(time.time()) & ((1 << 44) - 1)

        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]

        # CSV 저장
        if save_csv:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for group in range(GROUP_COUNT):
                    frequency_mhz = int(table[group])
                    writer.writerow([self.trial, group, frequency_mhz])

        self.trial += 1
        # if self.trial == 5000:
        #     self.trial = 0

        return final_freq


# === 메인 함수 ===
if __name__ == "__main__":
    a = Hopping_ver7()
    a.init_csv()

    start_time = time.perf_counter()
    for i in range(1000):
        b = a.final_table_gen(save_csv=True)
    end_time = time.perf_counter()

    # print(b)
    print(end_time - start_time)
