import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import chisquare
import sys

TYPE1 = "final-seq-AES"
TYPE2 = "final-seq-V3"
TYPE3 = "final-seq-V4"
TYPE4 = "final-seq-V5"
TYPE5 = "final-seq-V7"
TYPE6 = "final-seq-V6"



def load_data(type):
    file_path = f"test_result/{type}.csv"
    df = pd.read_csv(file_path)
    return df

def bin_frequency(freq, bin_size=0.2):
    """주파수를 지정된 bin_size 단위로 바이닝"""
    return np.floor(freq / bin_size) * bin_size

def bin_frequency_hz(freq, bin_size=225000):
    """주파수를 지정된 bin_size 단위로 바이닝 (Hz 단위)"""
    return np.floor(freq / bin_size) * bin_size
    
def analyze_frequency_distribution(df, type):
    # V7은 Hz 단위로 저장되어 있어서 별도 처리
    if type == TYPE5 or type == TYPE4 or type == TYPE2 or type == TYPE1:  # final-seq-V7
        # Hz 단위 주파수를 225000 단위로 바이닝
        binned_freqs = bin_frequency_hz(df["Frequency (MHz)"], 225000)  # 컬럼명은 동일하지만 실제로는 Hz 단위
        frequency_counts = binned_freqs.value_counts().sort_index()
        #frequency_counts = frequency_counts.drop(frequency_counts.index[888:])
        
        print(frequency_counts)
        
        bin_info = "1Hz"
        unit_info = "Hz"
        bin_size_display = "1Hz"
    else:
        # 0.2MHz 단위로 바이닝
        binned_freqs = bin_frequency(df["Frequency (MHz)"], 0.2)
        binned_freqs = binned_freqs.round(2)  # 부동소수점 오차 제거
        frequency_counts = binned_freqs.value_counts().sort_index()
        
        bin_info = "0.2MHz"
        unit_info = "MHz"
        bin_size_display = "0.2MHz"
    
    mean = frequency_counts.values.mean()
    std = frequency_counts.values.std()    
    
    # frequency_counts를 CSV 파일로 저장
    frequency_df = pd.DataFrame({
        f'Frequency Bin ({unit_info})': frequency_counts.index,
        'Count': frequency_counts.values
    })
    csv_filename = f'test_result/frequency_counts_{type}_binned_{bin_info}.csv'
    frequency_df.to_csv(csv_filename, index=False)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: 주파수 위치 히스토그램 (막대그래프)
    plt.subplot(2, 2, 1)  
    plt.bar(range(len(frequency_counts)), frequency_counts.values, color='blue')
    plt.title(f"Occurrence of Each Frequency Bin ({bin_size_display})-{type}")
    plt.xlabel("Frequency Bin Number")
    plt.ylabel("Occurrence of Each Frequency Bin")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: 주파수 위치 점-선 그래프
    plt.subplot(2, 2, 2)
    plt.plot(range(len(frequency_counts)), frequency_counts.values, 
             marker='o', markersize=3, linewidth=1, color='skyblue')
    plt.title(f"Frequency Occurrence Line Plot ({bin_size_display})-{type}")
    plt.xlabel("Frequency Bin Number")
    plt.ylabel("Occurrence of Each Frequency Bin")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 발생 횟수의 분포 (빈도값 기준 정규성 평가)
    counts_distribution = frequency_counts.value_counts().sort_index()

    # Plot 3: 동일한 발생 횟수를 가진 주파수 개수(주파수 위치 발생 빈도의 분포)
    plt.subplot(2, 2, 3)
    plt.bar(counts_distribution.index, counts_distribution.values, color='green', label='Frequency Distribution')
    plt.title(f"Distribution of Frequency Occurrence ({bin_size_display} bins)-{type}")
    plt.xlabel("Number of Occurrence")
    plt.ylabel("Number of Frequencies with same Occurrence")
    # mean, std 표시
    plt.text(
        0.98, 0.95,  # 오른쪽 위에 표시
        f"Mean: {mean:.2f}\nStd: {std:.2f}\nBin size: {bin_size_display}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    
    plt.tight_layout()
    plt.savefig(f'test_result/[{type}]analysis_binned_{bin_info}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return frequency_counts
    
def chi_square_uniformity_test(frequency_counts, bin_info="0.2MHz 바이닝"):
    # 관측된 빈도 (각 주파수별 발생 횟수)
    observed = frequency_counts.values
    
    # 기본 통계 정보
    total_observations = observed.sum()
    num_categories = len(observed)
    expected_freq = total_observations / num_categories
    expected = np.full(num_categories, expected_freq)
    
    print(f"총 관측값: {total_observations}")
    print(f"주파수 카테고리 수 ({bin_info}): {num_categories}")
    print(f"기댓값 (각 주파수 구간): {expected_freq:.2f}")
    print(f"관측값 범위: {observed.min()} ~ {observed.max()}")
    print(f"관측값 평균: {observed.mean():.2f}")
    print(f"관측값 표준편차: {observed.std():.2f}")
    
    # 카이제곱 검정 수행
    chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    
    print(f"Chi-square statistic: {chi_stat:.6f}")
    print(f"Degrees of freedom: {num_categories - 1}")
    
    # p-value 정밀 출력
    if p_value == 0.0:
        print("P-value: < 1e-300 (매우 작은 값)")
    elif p_value < 1e-10:
        print(f"P-value: {p_value:.2e}")
    else:
        print(f"P-value: {p_value:.10f}")
    
    # 결과 해석
    if p_value < 0.05:
        print("결론: 균등분포 가설 기각 (p < 0.05)")
        print("주파수 분포가 균등하지 않습니다.")
    else:
        print("결론: 균등분포 가설 기각할 수 없음 (p >= 0.05)")
        print("주파수 분포가 균등하다고 볼 수 있습니다.")

def test_hopseq(type):
    df = load_data(type)
    frequency_counts = analyze_frequency_distribution(df, type)
    
    # V7은 Hz 단위 바이닝
    if type == TYPE5:  # final-seq-V7
        bin_info = "225000Hz 바이닝"
    else:
        bin_info = "0.2MHz 바이닝"
    
    print(f"\n=== {type} 주파수 균등분포 검정 ({bin_info}) ===")
    chi_square_uniformity_test(frequency_counts, bin_info)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python test-hopseq.py <TYPE>")
        print("TYPE 옵션:")
        print("  AES  : final-seq-AES")
        print("  V3   : final-seq-V3")
        print("  V4   : final-seq-V4")
        print("  V5   : final-seq-V5")
        print("  V7   : final-seq-V7")
        print("  V6   : final-seq-V6")
        sys.exit(1)
    
    arg = sys.argv[1].upper()
    
    if arg == "AES":
        test_hopseq(TYPE1)
    elif arg == "V3":
        test_hopseq(TYPE2)
    elif arg == "V4":
        test_hopseq(TYPE3)
    elif arg == "V5":
        test_hopseq(TYPE4)
    elif arg == "V7":
        test_hopseq(TYPE5)
    elif arg == "V6":
        test_hopseq(TYPE6)
    else:
        print("잘못된 TYPE입니다. AES, V3, V4, V5, V7, V6 중 하나를 선택하세요.")
        sys.exit(1)