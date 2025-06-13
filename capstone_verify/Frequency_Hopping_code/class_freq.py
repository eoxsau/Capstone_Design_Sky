import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES

# === 상수 및 시스템 매개변수 정의 === (Hackrf 장비의 sample rate는 20MHz -> 기존 200MHz의 조건에서 1/10하여 사용 중)
GROUP_COUNT = 40
SLOT_BANDWIDTH = 22.5e3              # 슬롯 크기 (0.0225MHz = 22.5KHz)
GROUP_BANDWIDTH = 0.3825e6             # 그룹 하나당 대역폭 = 0.3825 MHz
MIN_SPACING = 10e3                   # 최소 이격 거리 = 10KHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2       # 황금비를 복소수 연산에 활용

TOTAL_BANDWIDTH = 20e6  # 전체 사용 가능한 대역폭
BLOCK_BANDWIDTH = GROUP_BANDWIDTH + MIN_SPACING  # 그룹 대역폭 + 최소 이격 거리 = 이격 거리까지 포함하여 하나의 그룹으로 해석
REST_BANDWIDTH = TOTAL_BANDWIDTH - BLOCK_BANDWIDTH * GROUP_COUNT

class Hopping_ver3:

    def __init__(self):
        self.c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
        self.trial = 0
        self.key = b'\x13\x6f\x95\x6e\xc6\x32\x20\x70\xc4\xb1\xd0\x73\x5b\x19\x29\x34\x0d\x9b\xaf\x32\x4a\xab\xe0\x46\x7e\xd4\xe4\x98\x17\x81\x09\x08'

    # === 복소수 위상 기반 정규화 ===
    def generate_complex_phase(self, z: complex, iterations: int = 7):
        for _ in range(iterations):
            phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)     # 위상값 계산
            rotator = np.exp(1j * phi)                          # 회전 연산자
            z = (z**2 + self.c) * rotator                            # 회전 및 복소수 제곱 연산
        angle = np.angle(z)                                     # 복소수의 위상
        norm = (angle + np.pi) / (2 * np.pi)                    # 위상을 0~1 범위로 정규화
        
        return norm

    # === 한번의 시드로 복소수 생성 ===
    def get_phase_from_aes(self, tod, label, offset=0):
        pt = f"{tod}-{self.trial}-{label}".encode()
        pt_hash = sha256(pt).digest()[:16]
        ct = AES.new(self.key, AES.MODE_ECB).encrypt(pt_hash)
        real = int.from_bytes(ct[offset:offset+4], 'big') / 1e9
        imag = int.from_bytes(ct[offset+4:offset+8], 'big') / 1e9
        return complex(real, imag)

    # === 도약 주파수 생성 함수 ===
    def generate_group_frequencies(self, tod):
        group_freqs = []

        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=0)
            norm = self.generate_complex_phase(z)
            base_freq = norm * TOTAL_BANDWIDTH

            # 최소 이격 거리 만족하는 위치 찾기
            attempts = 0
            while any(abs(base_freq - f) < (GROUP_BANDWIDTH + MIN_SPACING) for f in group_freqs):
                z += 0.001
                norm = self.generate_complex_phase(z)
                base_freq = norm * TOTAL_BANDWIDTH
                attempts += 1
                if attempts > 1000:
                    break

            group_freqs.append(base_freq)

        return group_freqs

    # === 그룹 순서 결정 함수 ===
    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=8)
            norm_list.append(self.generate_complex_phase(z))
        return list(np.argsort(norm_list))

    # === 최종 테이블 ===
    def final_table_gen(self):
        tod = int(time.time()) & ((1 << 44) - 1)

        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]

        self.trial += 1
        if self.trial == 5000:
            self.trial = 0

        return final_freq

# -------------------------------------------------------------- <구분선> --------------------------------------------------------------
class Hopping_ver5:

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
        real = int.from_bytes(ct[offset:offset+4], 'big') / 1e9
        imag = int.from_bytes(ct[offset+4:offset+8], 'big') / 1e9
        z = complex(real, imag)
        return z

    # === 도약 주파수 생성 함수 ===
    def generate_group_frequencies(self, tod):  # 주파수 위치 생성 함수
        group_freqs = []  # 그룹별 주파수 리스트

        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=0)
            norm = self.generate_complex_phase(z)
            centers_freq = norm * TOTAL_BANDWIDTH  # 중심 주파수 결정

            # 최소 이격 거리 만족하는 위치 찾기
            attempts = 0
            while any(abs(centers_freq - f) < (GROUP_BANDWIDTH + MIN_SPACING) for f in group_freqs):
                z += 0.001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
                norm = self.generate_complex_phase(z)
                centers_freq = norm * TOTAL_BANDWIDTH
                attempts += 1
                if attempts > 1000:  # 무한 루프 방지
                    break

            group_freqs.append(centers_freq)

        return group_freqs

    # === 그룹 순서 결정 함수 ===
    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_phase_from_aes(tod, gid, offset=8)
            norm_list.append(self.generate_complex_phase(z))
        return list(np.argsort(norm_list))

    # === 최종 테이블 ===
    def final_table_gen(self):
        tod = int(time.time()) & ((1 << 44) - 1)

        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]
        
        self.trial += 1
        if self.trial == 5000:
            self.trial = 0

        return final_freq

# -------------------------------------------------------------- <구분선> --------------------------------------------------------------
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

        # --- 대역폭에 맞게 조정 ---
        group_freqs = [freq * REST_BANDWIDTH for freq in group_freqs]
        
        # --- 이전 이격으로부터 블럭 대역폭에 다음 이격을 더하면 다음 주파수 위치 ---
        for k in range(1, GROUP_COUNT):
            group_freqs[k] = group_freqs[k] + group_freqs[k - 1] + BLOCK_BANDWIDTH

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

    # === 최종 테이블 ===
    def final_table_gen(self):
        tod = int(time.time()) & ((1 << 44) - 1)

        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)

        final_freq = [table[rank] for rank in order]

        self.trial += 1
        if self.trial == 5000:
            self.trial = 0

        return final_freq


# === 메인 함수 ===
if __name__ == "__main__":
    #a = Hopping_ver3()
    #b = Hopping_ver5()
    c = Hopping_ver7()
    #a.final_table_gen()
    #b.final_table_gen()
    c.final_table_gen()
    