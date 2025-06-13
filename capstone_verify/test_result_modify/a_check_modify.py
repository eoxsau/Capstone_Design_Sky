import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import chisquare
import sys

TYPE1 = "VerAES_modify"
TYPE2 = "Ver3_modify"
TYPE3 = "Ver4_modify"
TYPE4 = "Ver5_modify"
TYPE5 = "Ver7_modify"

def load_data(type):
    file_path = f"test_result_modify/{type}.csv"
    df = pd.read_csv(file_path)
    return df
    
def analyze_frequency_distribution(df, type):
    index_freqs = df["Frequency"]
    frequency_counts = index_freqs.value_counts().sort_index()

    bin_info = "Index"
    unit_info = "Hz"
    bin_size_display = "Index"
    
    mean = frequency_counts.values.mean()
    std = frequency_counts.values.std()    
    
    # frequency_counts를 CSV 파일로 저장
    frequency_df = pd.DataFrame({
        f'Frequency Bin ({unit_info})': frequency_counts.index,
        'Count': frequency_counts.values
    })
    csv_filename = f'test_result_modify/{type}_{bin_info}.csv'
    frequency_df.to_csv(csv_filename, index=False)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: 주파수 위치 히스토그램 (막대그래프)
    plt.subplot(2, 2, 1)  
    plt.bar(range(len(frequency_counts)), frequency_counts.values, color='blue')
    plt.title(f"Occurrence of Each Frequency Index ({bin_size_display})-{type}")
    plt.xlabel("Frequency Index Number")
    plt.ylabel("Occurrence of Each Frequency Index")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: 주파수 위치 점-선 그래프
    plt.subplot(2, 2, 2)
    plt.plot(range(len(frequency_counts)), frequency_counts.values, 
             marker='o', markersize=3, linewidth=1, color='skyblue')
    plt.title(f"Frequency Occurrence Line Plot ({bin_size_display})-{type}")
    plt.xlabel("Frequency Index Number")
    plt.ylabel("Occurrence of Each Frequency Index")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 발생 횟수의 분포 (빈도값 기준 정규성 평가)
    counts_distribution = frequency_counts.value_counts().sort_index()

    # Plot 3: 동일한 발생 횟수를 가진 주파수 개수(주파수 위치 발생 빈도의 분포)
    plt.subplot(2, 2, 3)
    plt.bar(counts_distribution.index, counts_distribution.values, color='green', label='Frequency Distribution')
    plt.title(f"Distribution of Frequency Occurrence ({bin_size_display})-{type}")
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
    plt.savefig(f'test_result_modify/[{type}]analysis_{bin_info}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return frequency_counts
    
def chi_square_uniformity_test(frequency_counts, bin_info="Index"):
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

    bin_info = "Index"
    print(f"\n=== {type} 주파수 균등분포 검정 ({bin_info}) ===")
    chi_square_uniformity_test(frequency_counts, bin_info)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python test-hopseq.py <TYPE>")
        print("TYPE 옵션:")
        print("  AES  : VerAES_modify")
        print("  V3   : Ver3_modify")
        print("  V4   : Ver4_modify")
        print("  V5   : Ver5_modify")
        print("  V7   : Ver7_modify")
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
    else:
        print("잘못된 TYPE입니다. AES, V3, V4, V5, V7 중 하나를 선택하세요.")
        sys.exit(1)