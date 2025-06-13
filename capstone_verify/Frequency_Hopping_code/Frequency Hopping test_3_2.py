import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from hashlib import sha256
from time import perf_counter

# === LEA 암호 클래스 ===
class LEA:
    delta = [0xc3efe9db, 0x44626b02, 0x79e27c8a, 0x78df30ec,
             0x715ea49e, 0xc785da0a, 0xe04ef22a, 0xe5c40957]

    def __init__(self, key: bytes):
        if len(key) != 16:
            raise ValueError("LEA 키는 128비트 (16바이트)만 지원합니다.")
        self.rounds = 24

        T = list(struct.unpack('<LLLL', key))

        self.rk = [[0] * 6 for _ in range(self.rounds)]

        for i in range(self.rounds):
            temp = self._rol(self.delta[i & 3], i)
            self.rk[i][0] = T[0] = self._rol((T[0] + temp) & 0xffffffff, 1)
            self.rk[i][1] = T[1] = self._rol((T[1] + self._rol(temp, 1)) & 0xffffffff, 3)
            self.rk[i][2] = T[2] = self._rol((T[2] + self._rol(temp, 2)) & 0xffffffff, 6)
            self.rk[i][3] = T[3] = self._rol((T[3] + self._rol(temp, 3)) & 0xffffffff, 11)
            self.rk[i][4] = self.rk[i][1]
            self.rk[i][5] = self.rk[i][1]

    def _rol(self, v, b):
        return ((v << b) | (v >> (32 - b))) & 0xffffffff

    def encrypt(self, pt: bytes) -> bytes:
        if len(pt) != 16:
            raise ValueError("평문은 반드시 16바이트여야 합니다.")
        x = list(struct.unpack('<LLLL', pt))
        for i in range(0, self.rounds, 4):
            # LEA의 라운드 연산
            x[3] = self._rol(((x[2] ^ self.rk[i][4]) + (x[3] ^ self.rk[i][5])) & 0xffffffff, 3)
            x[2] = self._rol(((x[1] ^ self.rk[i][2]) + (x[2] ^ self.rk[i][3])) & 0xffffffff, 5)
            x[1] = self._rol(((x[0] ^ self.rk[i][0]) + (x[1] ^ self.rk[i][1])) & 0xffffffff, 9)
            x[0] = self._rol(((x[3] ^ self.rk[i][4]) + (x[0] ^ self.rk[i][5])) & 0xffffffff, 3)
        return struct.pack('<LLLL', *x)

# === 상수 및 시스템 매개변수 정의 ===
GROUP_COUNT = 45  # 전체 그룹 수
SLOT_BANDWIDTH_MHZ = 0.225  # 슬롯 크기 (22.5 kHz → 0.0225MHz)
GROUP_BANDWIDTH_MHZ = SLOT_BANDWIDTH_MHZ * 17  # 그룹 하나당 대역폭 = 0.3825 MHz (0.0225 * 17 = 0.3825)
TOTAL_BANDWIDTH_MHZ = 200  # 전체 사용 가능한 대역폭
MIN_SPACING_MHZ = SLOT_BANDWIDTH_MHZ * 5  # 최소 이격 거리 = 슬롯 기준 5개 = 0.0225 * 17 = 0.1125 MHz
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # 황금비를 복소수 연산에 활용


# === 복소수 위상 기반 정규화 ===
def generate_complex_phase(z: complex, iterations: int = 7):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    for _ in range(iterations):
        phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)  # 위상값 계산 (phi를 이렇게 정의한 이유는 더욱 복잡하게 만들기 위해 z의 실수값과 황금비를 곱하고 그걸 0부터 1까지의 값으로 정규화함.)
        rotator = np.exp(1j * phi)  # 회전 연산자 (오일러 함수)
        z = (z**2 + c) * rotator  # 회전 및 복소수 제곱 연산
    angle = np.angle(z)  # 복소수의 위상 (역연산으로 정확한 복소수 좌표를 얻기 힘듦. 해당 좌표와 원점이 이루는 선상의 모든 점의 좌표가 경우의 수가 되기 때문)
    norm = (angle + np.pi) / (2 * np.pi)  # 위상을 0~1 범위로 정규화 (위상을 그대로 사용할 경우에 값이 매우 커질 수 있음.)
    return norm, angle

# === 8바이트 시드로 복소수 생성 ===
def get_complex_from_seed(seed: bytes):
    real = int.from_bytes(seed[:4], 'big') / 1e9  # 상위 4바이트 -> 실수 (10^9로 나누는 이유는 unsigned일 떄, 최대 2^32-1 = 4,294,967,295 의 값을 가지기 때문에 10^9을 통해 값을 줄임.)
    imag = int.from_bytes(seed[4:8], 'big') / 1e9  # 하위 4바이트 → 허수 (,,,)
    return complex(real, imag)

# === 주파수 시드 생성 함수 ===
def generate_frequency_seed(tod, trial, group, key):  # 주파수 위치를 정하는 복소수 연산에 필요한 시드 생성 함수
    pt = f"{tod}-{trial}-{group}".encode()  # 평문 구성 ( tod(송신 시각을 정밀하게 나타냄, 초 단위를 실수형으로 받아옴.) - 44비트 이내, 시도 횟수, 그룹 번호로 구성됨.f는 구분자)
    pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
    ct = LEA(key).encrypt(pt_hash)  # AES 암호화
    return get_complex_from_seed(ct)  # 초기 복소수 반환

# === 그룹 순서 시드 생성 함수 ===
def generate_group_order_seed(tod, trial, rank, key):  # 그룹 순서를 정하는 복소수 연산에 필요한 시드 생성 함수
    pt = f"{tod}-{trial}-R{rank}".encode()  # 순서용 평문 구성 ( tod(송신 시각을 정밀하게 나타냄) - 44비트, 시도 횟수, 랭크 값으로 구성됨. f, R은 구분자)
    pt_hash = sha256(pt).digest()[:16]  # 16바이트 해시값 생성
    ct = LEA(key).encrypt(pt_hash)  # AES 암호화
    return get_complex_from_seed(ct[8:])  # 8~15바이트 사용

# === 도약 주파수 생성 함수 ===
def generate_group_frequencies(tod, trial, key):  # 주파수 위치 생성 함수
    group_freqs = []  # 그룹별 주파수 리스트
    used_freqs = []  # 사용된 중심 주파수 저장용

    for group_id in range(GROUP_COUNT):
        z = generate_frequency_seed(tod, trial, group_id, key)  # 복소수 시드 생성
        norm, _ = generate_complex_phase(z)
        base_freq = norm * TOTAL_BANDWIDTH_MHZ  # 중심 주파수 결정
        """ 동기화 기준을 제공하기 위해서 중심 주파수를 기준으로 정해야함.
        수신기는 송신기의 주파수를 모르기 때문에 도약 알고리즘을 공유하되, 중심 주파수를 기준으로 해석해야 정확한 위치를 알 수 있음.
        상대 도약 (주파수 위치) 만 존재하면 시스템간 정렬이 불가능함."""

        # 최소 이격 거리 만족하는 위치 찾기
        attempts = 0
        while any(abs(base_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in used_freqs):
            """
            base_freq: 현재 계산된 주파수 후보 (복소수 위상 기반으로 생성)
            used_freqs: 이전에 이미 채택된 도약 주파수들 리스트 (같은 그룹 또는 인접 그룹)
            GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ: 허용되는 최소 간격
            GROUP_BANDWIDTH_MHZ: 한 그룹이 차지하는 주파수 폭
            MIN_SPACING_MHZ: 그 그룹 사이에 띄워야 하는 보호 간격
            """
            z += 0.000001  # z 값을 미세하게 변경하여 반복 (미세하게 z값을 조정하여 새 위상 기반 주파수를 생성하게함.)
            norm, _ = generate_complex_phase(z)
            base_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            if attempts > 1000:  # 무한 루프 방지
                break

        used_freqs.append(base_freq)
        slots = [base_freq + i * SLOT_BANDWIDTH_MHZ for i in range(17)]  # 슬롯 17개 주파수 생성
        group_freqs.append(slots)

    return group_freqs

# === 그룹 순서 결정 함수 ===
def generate_group_order(tod, trial, key):  # 그룹의 순서를 결정하는 함수
    scores = []
    for rank in range(GROUP_COUNT):
        z = generate_group_order_seed(tod, trial, rank, key)  # 그룹 순서를 결정하기 위한 복소수 시드 생성
        _, angle = generate_complex_phase(z)  # 위상값 계산
        scores.append((angle, rank))  # 위상값 기준 정렬 (크기로 정렬)
    scores.sort()
    return [rank for _, rank in scores]

# === 시각화 함수 ===
def visualize_all(trials=10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14), sharex=True)
    key = b'0123456789abcdef'  # 16바이트 LEA 키
    tod = int(time.time()) & ((1 << 44) - 1)  # TOD 값 (44비트 제한, Shift 연산과 and 연산으로 마스킹 하였음.)

    # --- 주파수 위치 시각화 ---
    for group_id in range(GROUP_COUNT):
        freqs = []
        for trial in range(trials):
            table = generate_group_frequencies(tod, trial, key)
            freqs.append(table[group_id][0])  # 각 그룹의 첫 슬롯만 시각화
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

# === 메인 함수 ===
if __name__ == "__main__":
    start = perf_counter()
    visualize_all(trials=10)
    end = perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")