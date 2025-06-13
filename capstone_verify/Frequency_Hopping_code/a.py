import numpy as np

# === 복소수 위상 기반 정규화 ===
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2

def generate_complex_phase(z):  # 반복 복소수 연산 함수
    c = complex(GOLDEN_RATIO, GOLDEN_RATIO)  # 반복에 사용될 복소수 상수
    
    phi = 2 * np.pi * ((z.real * GOLDEN_RATIO) % 1)
    rotator = np.exp(1j * phi)
    print(f"C = {c:.4f}")
    print(f"phi = {phi:.4f}")
    print(f"rotator = {rotator:.4f}")

    z = z**2
    print(f"z**2 = {z}")

    z = z + c
    print(f"z**2 + C = {z:.4f}")

    z = z * rotator
    print(f"(z**2 + C) * rotator = {z:.4f}")
    
    angle = np.angle(z)
    angle = np.rad2deg(angle)
    print(f"z's phase = {angle:.4f}")
    
    norm = (angle + np.pi) / (2 * np.pi)
    print(f"Normal = {norm:.4f}")

    return norm, angle

z = complex(87, 195)

generate_complex_phase(z)