import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import chisquare, chi2, kstest
import sys
import os

def load_data(filename):
    """CSV 파일을 로드"""
    if not os.path.exists(filename):
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        sys.exit(1)
    df = pd.read_csv(filename)
    return df

def analyze_frequency_distribution(df, filename):
    """888개 슬롯 기준 주파수 분포 분석"""
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # 888개 슬롯 기준 분석 (200MHz / 225KHz = 888.89 ≈ 888 슬롯)
    total_slots = 888  # 888 슬롯
    
    # CSV의 Slot 열에서 직접 슬롯 번호 읽기
    slot_numbers = df["Slot"].values
    
    # 슬롯별 발생 횟수 계산
    slot_counts = np.zeros(total_slots)
    for slot in slot_numbers:
        if 0 <= slot < total_slots:
            slot_counts[slot] += 1
    
    # 0이 아닌 슬롯만 추출하여 pandas Series로 변환
    non_zero_slots = []
    non_zero_counts = []
    for i, count in enumerate(slot_counts):
        if count > 0:
            non_zero_slots.append(i)
            non_zero_counts.append(count)
    
    frequency_counts = pd.Series(non_zero_counts, index=non_zero_slots)
    
    # 통계 계산
    mean = frequency_counts.values.mean()
    std = frequency_counts.values.std()
    
    # 결과를 CSV 파일로 저장
    frequency_df = pd.DataFrame({
        'Slot_Number': frequency_counts.index,
        'Count': frequency_counts.values
    })
    csv_filename = f'frequency_counts_{base_filename}_888slots.csv'
    frequency_df.to_csv(csv_filename, index=False)
    
    print(f"888개 슬롯 중 {len(non_zero_slots)}개 슬롯 사용됨")
    print(f"슬롯별 평균 발생 횟수: {mean:.2f}")
    print(f"표준편차: {std:.2f}")
    
    return frequency_counts, total_slots


def chi_square_uniformity_test(frequency_counts, total_slots):
    """888개 슬롯 기준 카이제곱 균등분포 검정"""
    observed = frequency_counts.values
    
    total_observations = observed.sum()
    used_slots = len(observed)
    expected_freq = total_observations / used_slots
    
    print(f"총 관측값: {total_observations}")
    print(f"사용된 슬롯 수: {used_slots}/{total_slots}")
    print(f"기댓값 (각 슬롯): {expected_freq:.2f}")
    
    # 카이제곱 통계량 계산
    chi2_stat = np.sum((observed - expected_freq) ** 2 / expected_freq)
    degrees_of_freedom = used_slots - 1
    p_value = chi2.sf(chi2_stat, degrees_of_freedom)
    
    print(f"\n=== Chi-square 균등분포 검정 ===")
    print(f"Chi-square statistic: {chi2_stat:.6f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    
    if p_value < 1e-10:
        print(f"P-value: {p_value:.2e}")
    else:
        print(f"P-value: {p_value:.10f}")
    
    if p_value < 0.05:
        print("결론: 균등분포 가설 기각 (p < 0.05)")
    else:
        print("결론: 균등분포 가설 채택 (p >= 0.05)")

def plot_combined_analysis(df, frequency_counts, filename, total_slots):
    """4개 그래프를 하나의 파일로 저장 (888개 슬롯 기준 분석)"""
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # 통계 계산
    mean = frequency_counts.values.mean()
    std = frequency_counts.values.std()
    counts_distribution = frequency_counts.value_counts().sort_index()
    
    # 0번째 그룹 데이터 추출
    group0_data = df[df['Group'] == 0].sort_values('Trial')
    max_trials = min(100, len(group0_data))
    trials = group0_data['Trial'].values[:max_trials]
    slot_numbers_group0 = group0_data['Slot'].values[:max_trials]
    
    # 4개 그래프를 하나의 파일로 생성
    plt.figure(figsize=(16, 12))
    
    # Plot 1: 슬롯별 발생 횟수 히스토그램
    plt.subplot(2, 2, 1)  
    plt.bar(frequency_counts.index, frequency_counts.values, color='blue')
    plt.title(f"Occurrence of Each Slot (888 Slots) - {base_filename}")
    plt.xlabel("Slot Number")
    plt.ylabel("Occurrence Count")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: 슬롯별 발생 횟수 선 그래프
    plt.subplot(2, 2, 2)
    plt.plot(frequency_counts.index, frequency_counts.values, 
             linewidth=2, color='skyblue', marker='o', markersize=2)
    plt.title(f"Slot Occurrence Line Plot - {base_filename}")
    plt.xlabel("Slot Number")
    plt.ylabel("Occurrence Count")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: 발생 횟수 분포 히스토그램
    plt.subplot(2, 2, 3)
    plt.bar(counts_distribution.index, counts_distribution.values, color='green', alpha=0.7)
    plt.title(f"Distribution of Occurrence Counts - {base_filename}")
    plt.xlabel("Occurrence Count")
    plt.ylabel("Number of Slots")
    # 통계 정보 표시
    plt.text(
        0.98, 0.95,
        f"Used Slots: {len(frequency_counts)}/{total_slots}\nMean: {mean:.2f}\nStd: {std:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 4: Group 0 슬롯 궤적
    plt.subplot(2, 2, 4)
    plt.plot(trials, slot_numbers_group0, 'r-', linewidth=1.5, marker='o', markersize=2)
    plt.title(f'Group 0 Slot Trajectory (First {max_trials} Trials)')
    plt.xlabel('Trial')
    plt.ylabel('Slot Number')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 통합 그래프 저장
    output_filename = f'[{base_filename}]_888slots_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"888개 슬롯 분석 그래프 저장: {output_filename}")

def autocorrelation_analysis(df, filename, max_lag=50):
    """각 그룹별 슬롯 위치의 자기상관계수 분석"""
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    print(f"\n=== 슬롯 기준 자기상관계수 분석 시작 ===")
    
    # 각 그룹별로 슬롯 시퀀스 추출
    group_sequences = {}
    autocorr_results = {}
    
    for group_id in range(40):  # 그룹 0부터 39까지
        group_data = df[df['Group'] == group_id].sort_values('Trial')
        
        # CSV의 Slot 열에서 직접 슬롯 번호 읽기
        slot_numbers = group_data['Slot'].values
        
        if len(slot_numbers) > max_lag:  # 충분한 데이터가 있는 경우만 분석
            group_sequences[group_id] = slot_numbers
            
            # 자기상관계수 계산
            autocorr = []
            n = len(slot_numbers)
            
            # 평균 제거 (중심화)
            slot_centered = slot_numbers - np.mean(slot_numbers)
            
            # lag 0부터 max_lag까지 자기상관계수 계산
            for lag in range(max_lag + 1):
                if lag == 0:
                    corr = 1.0  # lag 0은 항상 1
                else:
                    if n - lag > 0:
                        # 자기상관계수 계산
                        numerator = np.sum(slot_centered[:-lag] * slot_centered[lag:])
                        denominator = np.sum(slot_centered ** 2)
                        corr = numerator / denominator if denominator != 0 else 0
                    else:
                        corr = 0
                autocorr.append(corr)
            
            autocorr_results[group_id] = autocorr
    
    print(f"분석 완료: {len(autocorr_results)}개 그룹")
    
    # 그래프 생성 (4개 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Autocorrelation Analysis - {base_filename}', fontsize=16)
    
    # Plot 1: 그룹 0의 슬롯 기준 자기상관함수
    if 0 in autocorr_results:
        axes[0, 0].plot(range(max_lag + 1), autocorr_results[0], 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='±0.05 threshold')
        axes[0, 0].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Group 0 Slot Autocorrelation Function')
        axes[0, 0].set_xlabel('Lag')
        axes[0, 0].set_ylabel('Slot Autocorrelation')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # lag 1 값 표시
        if len(autocorr_results[0]) > 1:
            lag1_value = autocorr_results[0][1]
            axes[0, 0].annotate(f'Lag 1: {lag1_value:.4f}', 
                               xy=(1, lag1_value), xytext=(10, lag1_value + 0.1),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               fontsize=12, color='red', fontweight='bold')
    
    # Plot 2: 모든 그룹의 lag 1 자기상관계수
    lag1_values = []
    group_ids = []
    for group_id in sorted(autocorr_results.keys()):
        if len(autocorr_results[group_id]) > 1:
            lag1_values.append(autocorr_results[group_id][1])
            group_ids.append(group_id)
    
    axes[0, 1].bar(group_ids, lag1_values, color='skyblue', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='±0.05 threshold')
    axes[0, 1].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Lag 1 Slot Autocorrelation for All Groups')
    axes[0, 1].set_xlabel('Group ID')
    axes[0, 1].set_ylabel('Lag 1 Slot Autocorrelation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: 첫 5개 그룹의 자기상관함수 비교
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, group_id in enumerate(sorted(autocorr_results.keys())[:5]):
        axes[1, 0].plot(range(max_lag + 1), autocorr_results[group_id], 
                       color=colors[i], linewidth=1.5, label=f'Group {group_id}', alpha=0.8)
    
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Slot Autocorrelation Comparison (First 5 Groups)')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Slot Autocorrelation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: 자기상관계수 통계 히스토그램
    all_lag1_values = [autocorr_results[gid][1] for gid in autocorr_results.keys() if len(autocorr_results[gid]) > 1]
    axes[1, 1].hist(all_lag1_values, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(all_lag1_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_lag1_values):.4f}')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_title('Distribution of Lag 1 Slot Autocorrelation')
    axes[1, 1].set_xlabel('Lag 1 Slot Autocorrelation Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 그래프 저장
    output_filename = f'[{base_filename}]_autocorrelation_analysis.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과 출력
    print(f"\n=== 슬롯 기준 자기상관계수 분석 결과 ===")
    print(f"분석된 그룹 수: {len(autocorr_results)}")
    if all_lag1_values:
        print(f"Lag 1 슬롯 자기상관계수 통계:")
        print(f"  평균: {np.mean(all_lag1_values):.6f}")
        print(f"  표준편차: {np.std(all_lag1_values):.6f}")
        print(f"  최소값: {np.min(all_lag1_values):.6f}")
        print(f"  최대값: {np.max(all_lag1_values):.6f}")
        
        # 임계값 벗어나는 그룹 찾기
        threshold = 0.05
        problematic_groups = []
        for i, group_id in enumerate(group_ids):
            if abs(lag1_values[i]) > threshold:
                problematic_groups.append((group_id, lag1_values[i]))
        
        if problematic_groups:
            print(f"\n임계값 ±{threshold} 벗어나는 그룹:")
            for group_id, value in problematic_groups:
                print(f"  그룹 {group_id}: {value:.6f}")
        else:
            print(f"\n모든 그룹이 임계값 ±{threshold} 내에 있습니다.")
    
    print(f"슬롯 기준 자기상관계수 분석 그래프 저장: {output_filename}")
    
    return autocorr_results

def test_hopseq(filename):
    """888개 슬롯 기준 주파수 호핑 시퀀스 분석"""
    df = load_data(filename)
    
    # 888개 슬롯 기준 주파수 분포 분석
    frequency_data, total_slots = analyze_frequency_distribution(df, filename)
    
    # 4개 그래프를 하나의 파일로 저장
    plot_combined_analysis(df, frequency_data, filename, total_slots)
    
    # 슬롯 기준 자기상관계수 분석
    autocorr_data = autocorrelation_analysis(df, filename, max_lag=50)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    print(f"\n=== {base_filename} 888개 슬롯 균등분포 검정 ===")
    chi_square_uniformity_test(frequency_data, total_slots)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python test-hopseq.py <CSV파일명>")
        print("예시:")
        print("  python test-hopseq.py dynamic-final-seq-V5.csv")
        print("  python test-hopseq.py dynamic-final-seq-AES.csv")
        print("\n분석 내용:")
        print("  - 888개 슬롯 기준 주파수 분포 분석")
        print("  - 슬롯 기준 자기상관계수 분석")
        print("  - 균등분포 검정")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    test_hopseq(filename)