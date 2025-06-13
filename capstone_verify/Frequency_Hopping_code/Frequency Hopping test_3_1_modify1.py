import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 45  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.225  # 슬롯 크기 (225 kHz → 0.225MHz)
GROUP_BANDWIDTH_MHZ = 3.825  # 그룹 하나당 대역폭 = 3.825 MHz (0.225 * 17 = 3.825)
TOTAL_BANDWIDTH_MHZ = 200  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = 0.1  # 최소 이격 거리 (민간 항공에서 사용하는 50kHz의 이격보다 넓게 설정함.)
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용
key = b'0123456789abcdef0123456789abcdef'
tod = int(time.time()) & ((1 << 44) - 1)
trial = 0

# === 복소수 위상 기반 정규화 ===
def generate_complex_phase(z: complex, iterations: int = 2):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)  # 위상값 계산 (phi를 이렇게 정의한 이유는 더욱 복잡하게 만들기 위해 z의 실수값과 황금비를 곱하고 그걸 0부터 1까지의 값으로 정규화함.)
        rotator = np.exp(1j * phi)  # 회전 연산자 (오일러 함수)
        z = (z*z + c) * rotator  # 회전 및 복소수 제곱 연산
    angle = np.angle(z)  # 복소수의 위상 (역연산으로 정확한 복소수 좌표를 얻기 힘듦. 해당 좌표와 원점이 이루는 선상의 모든 점의 좌표가 경우의 수가 되기 때문)
    norm = (angle + np.pi) / (2 * np.pi)  # 위상을 0~1 범위로 정규화 (위상을 그대로 사용할 경우에 값이 매우 커질 수 있음.)
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
        z = get_phase_from_aes(tod, trial, gid, key, offset=0)
        norm = generate_complex_phase(z)
        centers_freq = norm * TOTAL_BANDWIDTH_MHZ  # 중심 주파수 결정
        """ 동기화 기준을 제공하기 위해서 중심 주파수를 기준으로 정해야함.
        수신기는 송신기의 주파수를 모르기 때문에 도약 알고리즘을 공유하되, 중심 주파수를 기준으로 해석해야 정확한 위치를 알 수 있음.
        상대 도약 (주파수 위치) 만 존재하면 시스템간 정렬이 불가능함."""

        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(centers_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            """
            base_freq: 현재 계산된 주파수 후보 (복소수 위상 기반으로 생성)
            used_freqs: 이전에 이미 채택된 도약 주파수들 리스트 (같은 그룹 또는 인접 그룹)
            GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ: 허용되는 최소 간격
            GROUP_BANDWIDTH_MHZ: 한 그룹이 차지하는 주파수 폭
            MIN_SPACING_MHZ: 그 그룹 사이에 띄워야 하는 보호 간격
            """
            z += 0.001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm = generate_complex_phase(z)
            centers_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break

        group_freqs.append(centers_freq)

    return group_freqs

# === 그룹 순서 결정 함수 ===
def generate_group_order(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=8)
        norm_list.append(generate_complex_phase(z))
    return list(np.argsort(norm_list))  # 크기 기준 정렬된 group_id 인덱스 반환

"""
# === 시각화 함수 ===
def visualize_all(trials=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14), sharex=True)
    key = b'0123456789abcdef0123456789abcdef'  # 32바이트 AES-256 키
    tod = int(time.time()) & ((1 << 44) - 1)  # TOD 값 (44비트 제한, Shift 연산과 and 연산으로 마스킹 하였음.)

    # --- 주파수 위치 시각화 ---
    for group_id in range(GROUP_COUNT):
        freqs = []
        for trial in range(trials):
            table = generate_group_frequencies(tod, trial, key)
            freqs.append(table[group_id])  # 각 그룹의 첫 슬롯만 시각화
        ax1.plot(range(trials), freqs, alpha=0.6)

    ax1.set_title("Trial-wise Frequency Location Transitions")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_ylim(0, TOTAL_BANDWIDTH_MHZ)
    ax1.grid(True)

    # --- 그룹 순서 시각화 ---
    group_history = [[] for _ in range(GROUP_COUNT)]
    for trial in range(trials):
        order = generate_group_order(tod, trial, key)
        for idx, gid in enumerate(order):
            group_history[gid].append(idx)

    for gid in range(GROUP_COUNT):
        ax2.plot(range(trials), group_history[gid], alpha=0.5)

    ax2.set_title("Trial-wise Group Order Transitions")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Rank")
    ax2.set_ylim(0, GROUP_COUNT)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
"""


# === 주파수 및 순서 테이블 통합 생성 ===
def generate_frequency_and_order_table(tod, trial, key):
    group_freqs = generate_group_frequencies(tod, trial, key)
    group_order = generate_group_order(tod, trial, key)

    return group_freqs, group_order

# === 메인 함수 ===
if __name__ == "__main__":
    start = perf_counter()
    freqs, order = generate_frequency_and_order_table(tod, trial, key)
    end = perf_counter()
    
    #print("중심 주파수 리스트:", freqs)
    #print("그룹 순서:", order)
    #visualize_all(trials=10)

    print(f"정밀 실행 시간: {end - start:.6f}초")