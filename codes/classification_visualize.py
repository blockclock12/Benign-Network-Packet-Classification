import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 업로드하신 파일의 경로와 이름을 사용합니다.
file_path = '/content/drive/MyDrive/ET-BERT-main/result/confusion_matrix'

try:
    # 1. 파일 내용 전체를 하나의 문자열로 읽어옵니다.
    with open(file_path, 'r') as f:
        raw_data = f.read()

    # 2. **오류 해결**: 문제의 원인인 '[' 와 ']' 문자를 공백으로 대체하여 숫자를 분리합니다.
    cleaned_data = raw_data.replace('[', ' ').replace(']', ' ')

    # 3. 정리된 문자열에서 모든 숫자를 추출하여 1차원 NumPy 배열로 만듭니다.
    #    separator를 공백으로 지정합니다.
    cm_flat = np.fromstring(cleaned_data, dtype=int, sep=' ')

    # 4. 1차원 배열을 120x120 행렬로 재구성합니다.
    n_classes = 120
    expected_size = n_classes * n_classes

    if cm_flat.size != expected_size:
        # 데이터 크기가 120x120이 아닌 경우, 가장 가까운 정사각형 크기로 조정하여 경고를 출력합니다.
        approx_size = int(np.sqrt(cm_flat.size))
        cm = cm_flat[:approx_size*approx_size].reshape((approx_size, approx_size))
        print(f"경고: 데이터의 총 요소 수({cm_flat.size})가 120x120이 아닙니다. {approx_size}x{approx_size} 크기로 조정하여 시각화합니다.")
    else:
        cm = cm_flat.reshape((n_classes, n_classes))


    # 5. Confusion Matrix 시각화 (Seaborn Heatmap)
    # 120x120 행렬에 적합하도록 figure size를 크게 설정합니다.
    plt.figure(figsize=(20, 20))

    # **Heatmap 설정:** annot=False로 설정하여 숫자 표시를 생략합니다.
    sns_plot = sns.heatmap(
        cm,
        cmap='viridis',
        annot=False,
        cbar=True
    )

    # 6. 플롯 가독성 조정
    # 120개의 틱 레이블을 모두 표시하면 겹치므로 제거합니다.
    sns_plot.set_xticklabels([])
    sns_plot.set_yticklabels([])
    sns_plot.tick_params(left=False, bottom=False)

    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.title(f'{cm.shape[0]}x{cm.shape[1]} Confusion Matrix Heatmap (Cleaned Data)', fontsize=20)

    # 플롯을 이미지 파일로 저장합니다.
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/ET-BERT-main/result/confusion_matrix_visualized.png')

    print("\n✅ 시각화가 완료되었습니다. 'confusion_matrix_visualized.png' 파일을 확인해주세요.")

except FileNotFoundError:
    print(f"❌ 오류: {file_path} 파일을 찾을 수 없습니다. 파일 경로를 다시 확인해주세요.")
