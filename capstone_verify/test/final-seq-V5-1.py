import numpy as np
import matplotlib.pyplot as plt
import time
from hashlib import sha256
from Crypto.Cipher import AES
from time import perf_counter
import csv
from datetime import datetime

# === 시스템 파라미터 ===
GROUP_COUNT = 40
SLOT_BANDWIDTH_MHZ = 0.225
GROUP_BANDWIDTH_MHZ = 3.825
TOTAL_BANDWIDTH_MHZ = 200
MIN_SPACING_MHZ = 0.1
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
c = complex(GOLDEN_RATIO, GOLDEN_RATIO)
AES_KEY = bytes.fromhex('136F956EC6322070C4B1D0735B1929340D9BAF324AABE0467ED4E49817810908')

USABLE_BANDWIDTH = TOTAL_BANDWIDTH_MHZ - ((GROUP_COUNT - 1) * (MIN_SPACING_MHZ + GROUP_BANDWIDTH_MHZ)) - GROUP_BANDWIDTH_MHZ - 0.1

# === 복소수 위상 반복 함수 ===
def generate_complex_phase(z: complex, iterations: int = 1):
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

# === AES 암호화 함수 ===
def get_phase_from_aes(tod, trial, label, key, offset=0):
    pt = f"{tod}-{trial}-{label}".encode()
    pt_hash = sha256(pt).digest()[:16]
    ct = AES.new(key, AES.MODE_ECB).encrypt(pt_hash)
    real = int.from_bytes(ct[offset:offset+4], 'big') / 1e9
    imag = int.from_bytes(ct[offset+4:offset+8], 'big') / 1e9
    z = complex(real, imag)
    return z

# === 실험용 주파수 생성 함수들 ===

# 버전 B: 순차 배치 + 개선된 충돌회피 (빈 공간 탐색)
def generate_group_frequencies_improved_avoid(tod, trial, key):
    group_freqs = []
    
    for group_id in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        centers_freq = norm * TOTAL_BANDWIDTH_MHZ

        # 충돌 시 빈 공간 찾기
        if any(abs(centers_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            # 기존 주파수들을 정렬
            sorted_freqs = sorted(group_freqs)
            
            # 첫 번째 간격 확인 (0 ~ 첫 번째 주파수)
            if len(sorted_freqs) > 0 and sorted_freqs[0] > GROUP_BANDWIDTH_MHZ:
                centers_freq = GROUP_BANDWIDTH_MHZ / 2
            else:
                # 기존 주파수들 사이의 간격 찾기
                found = False
                for i in range(len(sorted_freqs) - 1):
                    gap = sorted_freqs[i+1] - sorted_freqs[i]
                    if gap > (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) * 2:
                        centers_freq = sorted_freqs[i] + gap / 2
                        found = True
                        break
                
                # 마지막 주파수 이후 공간 확인
                if not found and len(sorted_freqs) > 0:
                    if TOTAL_BANDWIDTH_MHZ - sorted_freqs[-1] > GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ:
                        centers_freq = sorted_freqs[-1] + GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ

        group_freqs.append(centers_freq)

    return group_freqs

# === 도약 주파수 생성 함수 ===
def generate_group_frequencies(tod, trial, key):  # 주파수 위치 생성 함수
    initial_freqs = []  # 초기 랜덤 중심 주파수 리스트

    # 1. 랜덤 주파수 생성 (복소 위상 기반)
    for group_id in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        initial_freqs.append(norm)

    # 2. 정렬 (정규화된 0~1 값 기준)
    initial_freqs.sort()

    # 3. 전체 usable 대역 계산 (전체 - 이격 누적 - 마지막 그룹의 대역폭)
    # total_spacing = (GROUP_COUNT - 1) * MIN_SPACING_MHZ
    # usable_band = TOTAL_BANDWIDTH_MHZ - total_spacing - GROUP_BANDWIDTH_MHZ - 0.1

    # 4. 정규화된 값들을 usable_band 안으로 매핑
    scaled_freqs = [r * USABLE_BANDWIDTH for r in initial_freqs]

    #a = [scaled_freqs[0]]
    #for i in range(1, GROUP_COUNT):
    #    a.append(scaled_freqs[i] - scaled_freqs[i - 1])
    #print(a)

    #print(scaled_freqs)

    # 5. 이격 거리만큼 보정하여 최종 주파수 리스트 생성
    adjusted_freqs = [f + i * MIN_SPACING_MHZ for i, f in enumerate(scaled_freqs)]

    return adjusted_freqs

# 그룹 순서 결정
def generate_group_order(tod, trial, key):
    norm_list = []
    for gid in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, trial, gid, key, offset=8)
        norm_list.append(generate_complex_phase(z))
    return list(np.argsort(norm_list))

# === 초기 주파수 분포 분석 ===
def analyze_initial_distribution():
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)
    
    print("\n=== 초기 주파수 분포 분석 ===")
    
    # 여러 trial에서 초기 분포 수집
    initial_freqs = []
    for trial in range(100):  # 100개 trial 분석
        for group_id in range(GROUP_COUNT):
            z = get_phase_from_aes(tod, trial, group_id, key, offset=0)
            norm = generate_complex_phase(z)
            initial_freq = norm * TOTAL_BANDWIDTH_MHZ
            initial_freqs.append(initial_freq)
    
    # 구간별 분포 분석
    bins = [0, 50, 100, 150, 200]
    bin_labels = ['0-50', '50-100', '100-150', '150-200']
    
    for i in range(len(bins)-1):
        count = sum(1 for f in initial_freqs if bins[i] <= f < bins[i+1])
        percentage = (count / len(initial_freqs)) * 100
        print(f"{bin_labels[i]:>8} MHz: {count:4d}개 ({percentage:5.1f}%)")
    
    print(f"\n총 샘플: {len(initial_freqs)}개")
    print(f"평균: {np.mean(initial_freqs):.1f} MHz")
    print(f"표준편차: {np.std(initial_freqs):.1f} MHz")
    
    # 첫 번째 trial의 첫 10개 그룹 상세 분석
    print(f"\n=== 첫 번째 Trial 초기 10개 그룹 분석 ===")
    for group_id in range(10):
        z = get_phase_from_aes(tod, 0, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        initial_freq = norm * TOTAL_BANDWIDTH_MHZ
        
        print(f"그룹 {group_id}: norm={norm:.4f} → 주파수={initial_freq:6.1f} MHz")
    
    return initial_freqs

# === 양 끝 집중 현상 분석 ===
def analyze_why_edges():
    print("\n=== 양 끝 집중 현상 분석 ===")
    
    # 충돌 회피 메커니즘 시뮬레이션
    print("충돌 회피 과정에서 z 값 변화:")
    z_initial = complex(0.5, 0.5)  # 예시 초기값
    
    for i in range(10):
        norm = generate_complex_phase(z_initial)
        freq = norm * TOTAL_BANDWIDTH_MHZ
        print(f"시도 {i:2d}: z={z_initial.real:.4f}+{z_initial.imag:.4f}j → norm={norm:.4f} → freq={freq:6.1f}")
        z_initial += 0.001
    
    print(f"\nz += 0.001 조정의 특성:")
    print(f"- z 값이 계속 증가하면서 복소수 위상이 변화")
    print(f"- generate_complex_phase의 비선형 특성으로 인해 norm 값이 불규칙하게 변화")
    print(f"- 하지만 충분히 많이 조정하면 결국 0 또는 1에 가까운 값으로 수렴")
    print(f"- 이는 주파수로 변환시 0MHz 또는 200MHz 근처가 됨")

# === 주파수 할당 시스템 함수 ===
def analyze_frequency_distribution():
    # 시스템 파라미터 분석
    required_space_per_group = GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ
    total_required_space = GROUP_COUNT * required_space_per_group
    available_space = TOTAL_BANDWIDTH_MHZ
    
    print("=== 주파수 할당 시스템 분석 ===")
    print(f"그룹 수: {GROUP_COUNT}")
    print(f"그룹당 대역폭: {GROUP_BANDWIDTH_MHZ} MHz")
    print(f"최소 간격: {MIN_SPACING_MHZ} MHz")
    print(f"그룹당 필요 공간: {required_space_per_group} MHz")
    print(f"총 필요 공간: {total_required_space} MHz")
    print(f"사용 가능 공간: {available_space} MHz")
    print(f"여유 공간: {available_space - total_required_space} MHz")
    print(f"공간 활용률: {(total_required_space/available_space)*100:.1f}%")
    
    if total_required_space > available_space:
        print("⚠️  경고: 필요 공간이 사용 가능 공간을 초과합니다!")
        print("   이는 주파수가 양 끝으로 몰리는 주요 원인입니다.")
    
    return total_required_space <= available_space

# === 단일 Trial 디버깅 ===
def debug_single_trial():
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)
    
    print("\n=== 단일 Trial 디버깅 ===")
    group_freqs = []
    conflict_counts = []
    
    for group_id in range(GROUP_COUNT):
        z = get_phase_from_aes(tod, 0, group_id, key, offset=0)
        norm = generate_complex_phase(z)
        initial_freq = norm * TOTAL_BANDWIDTH_MHZ
        
        centers_freq = initial_freq
        attempts = 0
        conflicts = 0
        
        while any(abs(centers_freq - f) < (GROUP_BANDWIDTH_MHZ + MIN_SPACING_MHZ) for f in group_freqs):
            z += 0.001
            norm = generate_complex_phase(z)
            centers_freq = norm * TOTAL_BANDWIDTH_MHZ
            attempts += 1
            conflicts += 1
            if attempts > 1000:
                break
        
        group_freqs.append(centers_freq)
        conflict_counts.append(conflicts)
        
        print(f"그룹 {group_id:2d}: 초기={initial_freq:6.1f} → 최종={centers_freq:6.1f} (충돌:{conflicts:3d}회)")
    
    print(f"\n평균 충돌 횟수: {np.mean(conflict_counts):.1f}")
    print(f"최대 충돌 횟수: {max(conflict_counts)}")
    print(f"충돌 없이 배치된 그룹: {conflict_counts.count(0)}개")
    
    # 주파수 간격 분석
    sorted_freqs = sorted(group_freqs)
    gaps = [sorted_freqs[i+1] - sorted_freqs[i] for i in range(len(sorted_freqs)-1)]
    
    print(f"\n주파수 간격 통계:")
    print(f"평균 간격: {np.mean(gaps):.2f} MHz")
    print(f"최소 간격: {min(gaps):.2f} MHz")
    print(f"최대 간격: {max(gaps):.2f} MHz")
    
    return group_freqs, conflict_counts

# === 주파수 분포 히스토그램 함수 ===
def plot_frequency_histogram(final_table):
    all_freqs = [freq for trial in final_table for freq in trial]
    
    bin_width = 0.05
    bins = np.arange(0, TOTAL_BANDWIDTH_MHZ + bin_width, bin_width)
    
    plt.figure(figsize=(12, 6))
    plt.hist(all_freqs, bins=bins, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    plt.title('Frequency Distribution Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('Frequency Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 추가
    plt.text(0.02, 0.98, f'Total Samples: {len(all_freqs)}\nMean: {np.mean(all_freqs):.1f} MHz\nStd: {np.std(all_freqs):.1f} MHz', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frequency_histogram_V5_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"히스토그램 저장: {filename}")
    

# === 최종 테이블 ===
def final_table_gen(trials, save_csv=True, filename="final-seq-V5.csv"):
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)
    z = complex(tod, trials)
    # --- 최종 테이블 ---
    final_table = []
    for trial in range(trials):
        table = generate_group_frequencies(tod, trial, key)
        order = generate_group_order(tod, trial, key)
        final_freq = [table[rank] for rank in order]
        final_table.append(final_freq)
    
    # CSV 파일로 저장
    if save_csv:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow(['Trial', 'Group', 'Frequency (MHz)'])
            
            # 데이터 작성
            for trial in range(trials):
                for group in range(GROUP_COUNT):
                    frequency = round(final_table[trial][group], 3)
                    writer.writerow([trial, group, frequency])
    
    return final_table

# === 실험 비교 함수 ===
def compare_methods():
    key = AES_KEY
    tod = int(time.time()) & ((1 << 44) - 1)
    trials = 1000
    
    print("\n=== 방법별 효과 비교 실험 ===")
    
    methods = {
        "원본 (순차+기존충돌회피)": generate_group_frequencies,
        "버전B (순차+개선된충돌회피)": generate_group_frequencies_improved_avoid,
    }
    
    for name, method in methods.items():
        print(f"\n--- {name} ---")
        all_freqs = []
        
        for trial in range(trials):
            freqs = method(tod, trial, key)
            all_freqs.extend(freqs)
        
        # 양 끝 집중도 분석 (0-20MHz, 180-200MHz)
        edge_count = sum(1 for f in all_freqs if f < 20 or f > 180)
        edge_percentage = (edge_count / len(all_freqs)) * 100
        
        # 분포 균등성 (표준편차)
        std_dev = np.std(all_freqs)
        
        print(f"  양 끝 집중도 (0-20, 180-200MHz): {edge_count}/{len(all_freqs)} ({edge_percentage:.1f}%)")
        print(f"  전체 분포 표준편차: {std_dev:.1f} MHz")
        print(f"  평균 주파수: {np.mean(all_freqs):.1f} MHz")

# === 메인 함수 ===
if __name__ == "__main__":
    # 시스템 분석 실행
    # analyze_frequency_distribution()
    # analyze_initial_distribution()
    # analyze_why_edges()
    # debug_single_trial()

    print("\n" + "="*50)
    start = time.perf_counter()
    final_table = final_table_gen(1)  # 빠른 분석을 위해 1000으로 줄임
    end = time.perf_counter()
    print(f"정밀 실행 시간: {end - start:.6f}초")
    
    # # 주파수 히스토그램 생성
    # plot_frequency_histogram(final_table)
    
    # # 실험 비교 실행
    # compare_methods()