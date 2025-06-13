import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256

# ---------- LEA Class ---------- (수정 해야 할껄, 그냥 가져다 쓴겁니당,, ㅋㅋㅋㅋㅋ)

SIZE_128 = 16
SIZE_192 = 24
SIZE_256 = 32
block_size = 16

class LEA:
    delta = [0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
             0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957]

    def __init__(self, key):
        mk = bytearray(key)
        key_size = len(mk)
        if key_size == SIZE_128:
            self.rounds = 24
        elif key_size == SIZE_192:
            self.rounds = 28
        elif key_size == SIZE_256:
            self.rounds = 32
        else:
            raise ValueError("Invalid key length")

        self.rk = [[0]*6 for _ in range(self.rounds)]
        T = [0]*8
        T[0], T[1], T[2], T[3] = struct.unpack('<LLLL', mk[:16])
        if key_size == SIZE_128:
            for i in range(self.rounds):
                temp = self.rol(self.delta[i & 3], i)
                self.rk[i][0] = T[0] = self.rol((T[0] + temp) & 0xffffffff, 1)
                self.rk[i][1] = T[1] = self.rol((T[1] + self.rol(temp, 1)) & 0xffffffff, 3)
                self.rk[i][2] = T[2] = self.rol((T[2] + self.rol(temp, 2)) & 0xffffffff, 6)
                self.rk[i][3] = T[3] = self.rol((T[3] + self.rol(temp, 3)) & 0xffffffff, 11)
                self.rk[i][4] = self.rk[i][1]
                self.rk[i][5] = self.rk[i][1]

    def rol(self, v, b):
        return ((v << b) | (v >> (32 - b))) & 0xffffffff

    def encrypt(self, pt):
        if len(pt) != 16:
            raise ValueError("Plaintext must be 16 bytes")
        x = list(struct.unpack('<LLLL', pt))
        for i in range(0, self.rounds, 4):
            x[3] = self.rol(((x[2] ^ self.rk[i][4]) + (x[3] ^ self.rk[i][5])) & 0xffffffff, 3)
            x[2] = self.rol(((x[1] ^ self.rk[i][2]) + (x[2] ^ self.rk[i][3])) & 0xffffffff, 5)
            x[1] = self.rol(((x[0] ^ self.rk[i][0]) + (x[1] ^ self.rk[i][1])) & 0xffffffff, 9)
            x[0] = self.rol(((x[3] ^ self.rk[i][4]) + (x[0] ^ self.rk[i][5])) & 0xffffffff, 3)
        return struct.pack('<LLLL', *x)

# ---------- ECB Mode ----------
class ECB:
    def __init__(self, cipher):
        self.cipher = cipher

    def encrypt_block(self, block):
        return self.cipher.encrypt(block)

# ---------- 복소수 LEA 기반 시드 생성기 ----------
def generate_complex_seed_lea(tod: int, key: bytes, trial: int, group: int, offset: int) -> complex:
    """
    TOD, Key, Trial, Group, Offset을 시드로 사용해 복소수 기반 난수 생성
    tod: 44비트 시간 정보, key: 암호화 키, trial: 시도 번호, group: 그룹 번호, offset: 슬롯 오프셋
    """
    pt = (f"{tod:044b}-{trial}-{group}-{offset}").encode()  # 시드 문자열 생성
    pt = sha256(pt).digest()[:16]   # SHA-256 해시 후 16바이트 사용
    cipher = LEA(key)   # LEA 암호화 객체 생성
    ecb = ECB(cipher)   # ECB 암호화 모드 설정
    ct = ecb.encrypt_block(pt)
    real = int.from_bytes(ct[0:4], 'big') / 1e9   # 실수값 정규화 (암호화된 비트가 크기 때문에 복소수로 사용하기 위해서 크기를 줄임)
    imag = int.from_bytes(ct[4:8], 'big') / 1e9   # 허수값 정규화 (,,)
    return complex(real, imag)  # 복소수로 반환

# ---------- 도약 조건 정의 ----------
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비는 (√5 - 1) / 2, 그냥 이뻐서 쓴 거

SLOTS_PER_GROUP = 17        # 각 그룹에 포함될 슬롯 수 (한 그룹에 17개의 슬롯)
GROUP_COUNT = 10            # 전체 그룹 수 (총 45개의 그룹을 설정
TOTAL_SLOTS = 8888          # 전체 슬롯 수 (200 MHz 대역폭에 해당하는 슬롯 수)
SLOT_BANDWIDTH_HZ = 22500   # 한 슬롯이 차지하는 대역폭 (22.5 kHz)
SLOT_BANDWIDTH_MHZ = SLOT_BANDWIDTH_HZ / 1e6  # 한 슬롯의 대역폭을 MHz로 변환 (22500 Hz -> 0.0225 MHz)
TOTAL_BANDWIDTH_HZ = TOTAL_SLOTS * SLOT_BANDWIDTH_HZ    # 전체 대역폭 (슬롯 수 * 한 슬롯 대역폭, 단위: Hz)
TOTAL_BANDWIDTH_MHZ = TOTAL_BANDWIDTH_HZ / 1e6  # 전체 대역폭을 MHz로 변환 (주파수 범위, 단위: MHz)
MIN_SPACING_MHZ = SLOT_BANDWIDTH_MHZ * 5  # 최소 간격을 슬롯 폭의 5배로 설정 (슬롯 간 최소 이격, 단위: MHz)

# ---------- 복소수 위상 반복을 통한 주파수 생성 ----------
def generate_complex_phase(z):
    """
    복소수 위상 반복을 사용해 주파수 계산
    z: 복소수 시드
    """
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)     # 황금비를 사용해 복소수 상수 c 정의 (그냥 반복되지 않는 무리수를 연산하면 그냥 실수보다 괜찮을 것 같아서 이쁜거로 했음)
    for _ in range(3):  # 3번 반복 (도약 당 3번)
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)   # 위상 계산
        rotator = np.exp(1j * phi)      # 회전 연산 (e^(j*phi))
        z = (z ** 2 + c) * rotator      # 복소수 회전 및 위상 변화
    angle = np.angle(z)     # 위상값 추출
    norm = (angle + np.pi) / (2 * np.pi)    # 0~1로 정규화
    return norm * TOTAL_BANDWIDTH_MHZ, angle    # 주파수 계산

# ---------- 대역 초과 시 wrap-around 처리 ----------
"""
    주파수 위치가 대역폭을 벗어나지 않도록 처리
    freq: 계산된 주파수 위치
    """
def wrap_frequency(freq):
    freq = freq % TOTAL_BANDWIDTH_MHZ   # 대역폭 범위 안으로 들어오도록 조절
    if freq + SLOT_BANDWIDTH_MHZ > TOTAL_BANDWIDTH_MHZ:
        freq = TOTAL_BANDWIDTH_MHZ - SLOT_BANDWIDTH_MHZ     # 슬롯 폭을 고려해서 구역 안으로 위치하도록 조정
    return freq

# ---------- 그룹별 주파수 도약 테이블 생성 ----------
"""
    주파수 도약 테이블 생성 (그룹별 주파수 위치 배치)
    trial: 시도 번호, tod: 시간, key: 암호화 키
    """
def generate_hopping_table(trial, tod, key):
    group_table = []    # 그룹별 주파수 리스트 초기화
    for group_id in range(GROUP_COUNT):
        group_freqs = []    # 각 그룹에 할당될 주파수 리스트
        offset = 0  # 슬롯 오프셋 초기화
        while len(group_freqs) < SLOTS_PER_GROUP and offset < 1000:
            z = generate_complex_seed_lea(tod, key, trial, group_id, offset)    # 복소수 난수 생성
            freq, _ = generate_complex_phase(z)   # 주파수 계산 (주파수 도약 순서를 나타낼 때 랭크값 계산 시 사용하려고 '_'를 남겨두었음)
            freq = wrap_frequency(freq)   # 대역폭 초과 시 다시 들어오게 하기
            if all(abs(freq - f) >= MIN_SPACING_MHZ + SLOT_BANDWIDTH_MHZ for f in group_freqs):
                group_freqs.append(freq)  # 겹치지 않으면 주파수 추가
            offset += 1 # 겹치면 오프셋 증가 (다음 슬롯 위치 계산을 위해 오프셋 증가)
        group_table.append(group_freqs) # 각 그룹에 대한 주파수 위치 테이블 생성
    return group_table

# ---------- 시도별 위치 변화를 시각화 ----------
def visualize_location_transitions_all(trials=10):
    """
    여러 시도(trials) 동안 주파수 도약 위치 변화 시각화
    trials: 시도 횟수 (기본값 10)
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    tod = int(time.time()) & ((1 << 44) - 1)  # TOD 값 계산 (44비트 시간)
    key = b"0123456789abcdef"  # 고정된 키

    for group_id in range(GROUP_COUNT): # 그룹별로 반복
        freqs_per_slot = [[] for _ in range(SLOTS_PER_GROUP)]  # 슬롯별 도약 위치 기록

        for trial in range(trials):
            table = generate_hopping_table(trial, tod, key)
            group = table[group_id]
            for slot_idx in range(len(group)):
                freqs_per_slot[slot_idx].append(group[slot_idx])

        # 슬롯 별 주파수 이동 시각화
        for slot_idx, freqs in enumerate(freqs_per_slot):
            if len(freqs) == trials:
                ax.plot(range(trials), freqs, alpha=0.4, linewidth=0.7)

    ax.set_title("Trial-wise Frequency Location Transitions (LEA-based Seed)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_ylim(0, TOTAL_BANDWIDTH_MHZ)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- 실행 ----------
if __name__ == "__main__":
    visualize_location_transitions_all(trials=10)