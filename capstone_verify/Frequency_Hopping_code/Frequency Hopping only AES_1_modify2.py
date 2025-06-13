import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 40  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = SLOT_BANDWIDTH_MHZ * 17
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.1  # 최소 이격 거리
AES_KEY = b'0123456789abcdef0123456789abcdef'  # AES-256 키

def aes_encrypt_128bit_int(data: bytes, key: bytes) -> int:
        cipher = AES.new(key, AES.MODE_ECB)
        ct = cipher.encrypt(data)
        return int.from_bytes(ct, 'big')

def generate_group_frequencies(tod, trial, key):
        freqs = []
        value_6bit_slice = 0
        
        pt = f"{tod}-{trial}-{GROUP_COUNT}".encode()
        pt_hash = sha256(pt).digest()[:16]
        value = aes_encrypt_128bit_int(pt_hash, key)
        
        while (value == 0):
            value_6bit_slice ^= value & '\x3F'
            value >>= 6
            
        freqs.append(freq)
        return freqs

# === 그룹 순서 생성 함수 ===
# === 시각화 함수 ===

# === 메인 함수 ===
if __name__ == "__main__":
    tod = int(time.time()) & ((1 << 44) - 1)
    trial = 0
    result = generate_group_frequencies(tod, trial, AES_KEY)
    for i, f in enumerate(result):
        print(f"Group {i:02d}: {f:.6f} MHz")