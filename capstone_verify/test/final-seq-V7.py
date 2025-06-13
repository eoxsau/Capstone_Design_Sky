import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES
import csv

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_COUNT = 17
SLOT_BANDWIDTH = 225e3  # 슬롯 크기 (0.225MHz = 225KHz)
GROUP_BANDWIDTH = 3.825e6  # 그룹 하나당 대역폭 (3.825MHz = 3825KHz)
MIN_SPACING = 100e3  # 최소 이격 거리 = 100KHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

TOTAL_BANDWIDTH = 200e6  # 전체 사용 가능한 대역폭
REST_BANDWIDTH = TOTAL_BANDWIDTH - (GROUP_BANDWIDTH * GROUP_COUNT) - (MIN_SPACING * (GROUP_COUNT - 1))  # 전체 대역폭에서 실제 사용되는 전체 대역폭을 제외한 나머지 대역폭 (우리의 예제는 200 - 3.925 * 40 = 43MHz 남음)

BLOCK_BANDWIDTH = GROUP_BANDWIDTH + MIN_SPACING
USABLE_BANDWIDTH_222 = (TOTAL_BANDWIDTH - ((GROUP_COUNT - 1) * BLOCK_BANDWIDTH) - GROUP_BANDWIDTH) / 2

TOTAL_SLOT_INDEX = int(TOTAL_BANDWIDTH / SLOT_BANDWIDTH)
USABLE_SLOT = TOTAL_SLOT_INDEX - SLOT_COUNT
REST_SLOT = USABLE_SLOT - (GROUP_COUNT * SLOT_COUNT) - GROUP_COUNT

class Hopping_ver7:

    def __init__(self):
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
            z = (z * z + c) * rotator * rotator_zc

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
        scale_freqs = [freq / error_sum for freq in group_freqs]

        #group_freqs.sort()

        # --- 대역폭에 맞게 조정 ---
        scale2_freqs = [int(freq * REST_BANDWIDTH) for freq in scale_freqs]

        for k in range(1, GROUP_COUNT):
            scale2_freqs[k] += MIN_SPACING

        # --- 이전 이격으로부터 블럭 대역폭에 다음 이격을 더하면 다음 주파수 위치 ---
        a = [scale2_freqs[0]]
        for k in range(1, GROUP_COUNT):
            a.append(scale2_freqs[k] + a[k - 1] + GROUP_BANDWIDTH)

        # --- 마지막 이격 정보는 필요없음 ---
        #group_freqs.pop()
        #print(max(a))

        return a

    def generate_group_frequencies_modify(self, tod):
        z = self.get_phase_from_aes(tod, 0, offset=0)
        norm = self.generate_complex_phase(z)
        start_pos = norm * GROUP_BANDWIDTH

        group_freqs = []
        for gid in range(1, GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=0)
            norm = self.generate_complex_phase(z)
            group_freqs.append(norm)
            
        error_sum = sum(group_freqs) / USABLE_BANDWIDTH_222
        group_freqs = [freq / error_sum for freq in group_freqs]

        group_freqs.insert(0, start_pos)
        for i in range(1, GROUP_COUNT):
            group_freqs[i] = group_freqs[i] + group_freqs[i - 1] + GROUP_BANDWIDTH + MIN_SPACING
            group_freqs[i] %= TOTAL_BANDWIDTH

        return group_freqs
    
    def generate_group_frequencies_modify_2(self, tod):
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

        print(max(group_freqs))

        return group_freqs

    # === 그룹 순서 결정 함수 ===
    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=8)
            norm_list.append(self.generate_complex_phase(z))
        return list(np.argsort(norm_list))

    def init_csv(self, filename="test_result/final-seq-V7.csv"):
        """CSV 파일 초기화 및 헤더 작성"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Trial', 'Group', 'Frequency (MHz)'])

    # === 최종 테이블 ===
    def final_table_gen(self, save_csv=False, filename="test_result/final-seq-V7.csv"):
        tod = int(time.time()) & ((1 << 44) - 1)

        #table = self.generate_group_frequencies(tod)
        table = self.generate_group_frequencies_modify_2(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]

        # CSV 저장
        if save_csv:
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for group in range(GROUP_COUNT):
                    frequency_mhz = int(final_freq[group])
                    writer.writerow([self.trial, group, frequency_mhz])

        self.trial += 1
        # if self.trial == 5000:
        #     self.trial = 0

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
    print("CSV 파일 저장 완료: final-seq-V7.csv")
