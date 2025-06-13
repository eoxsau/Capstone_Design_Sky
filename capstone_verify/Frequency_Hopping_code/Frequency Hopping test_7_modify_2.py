import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40
SLOT_BANDWIDTH = 225e3
GROUP_BANDWIDTH = 3.825e6
MIN_SPACING = 100e3
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)

TOTAL_BANDWIDTH = 200e6
BLOCK_BANDWIDTH = GROUP_BANDWIDTH + MIN_SPACING
REST_BANDWIDTH = TOTAL_BANDWIDTH - BLOCK_BANDWIDTH * GROUP_COUNT


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
        return z

    def get_z_from_tod(self, tod, gid_offset=0):
        real = ((tod >> 22) & 0x3FFFFF) / 1e6
        imag = (tod & 0x3FFFFF) / 1e6
        base_z = complex(real, imag)
        return self.generate_complex_phase(base_z + complex(self.trial, gid_offset))

    def aes_encrypt(self, z: complex):
        pt = sha256(str(z).encode()).digest()[:16]
        cipher = AES.new(self.key, AES.MODE_ECB)
        return cipher.encrypt(pt)

    def generate_group_frequencies(self, tod):
        rest_freqs = []
        group_freqs = []

        for gid in range(GROUP_COUNT + 1):
            z = self.get_z_from_tod(tod, gid_offset=gid)
            ct = self.aes_encrypt(z)
            # 상위 64비트 → 주파수용
            val_freq = int.from_bytes(ct[:8], 'big') / (2 ** 64 - 1)
            # 하위 64비트 → 그룹 순서용
            group_freqs.append(val_freq)

        error_sum = sum(group_freqs)
        group_freqs = [f / error_sum for f in group_freqs]
        group_freqs = [f * REST_BANDWIDTH for f in group_freqs]

        for k in range(1, GROUP_COUNT):
            group_freqs[k] = group_freqs[k] + group_freqs[k - 1] + BLOCK_BANDWIDTH

        group_freqs.pop()
        return group_freqs

    def generate_group_order(self, tod):
        norm_list = []
        for gid in range(GROUP_COUNT):
            z = self.get_z_from_tod(tod, gid_offset=gid)
            ct = self.aes_encrypt(z)
            score = int.from_bytes(ct[:8], 'big') / (2 ** 64 - 1)
            norm_list.append(score)
        return list(np.argsort(norm_list))

    def final_table_gen(self):
        tod = int(time.time()) & ((1 << 44) - 1)
        table = self.generate_group_frequencies(tod)
        order = self.generate_group_order(tod)
        final_freq = [table[rank] for rank in order]
        self.trial += 1
        return final_freq


# === 메인 함수 ===
if __name__ == "__main__":
    a = Hopping_ver7()
    start_time = time.perf_counter()
    for i in range(100):
        b = a.final_table_gen()
    end_time = time.perf_counter()
    print(end_time - start_time)