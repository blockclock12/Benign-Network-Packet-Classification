import pandas as pd

# 경로
MERGED_PATH = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/merged_etbert_dataset.tsv"
PACKET_PATH = "/content/packet_dataset.tsv"   # 사용자가 업로드한 파일
OUTPUT_PATH = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/merged_final_clean.tsv"

# 1) 기존 데이터 불러오기
df_main = pd.read_csv(MERGED_PATH, sep="\t", low_memory=False)

# 2) 업로드한 패킷 데이터 불러오기
df_pkt = pd.read_csv(PACKET_PATH, sep="\t", low_memory=False)

# 3) 업로드한 데이터에 label_bin 생성 (기존 label을 그대로 사용)
if "label_bin" not in df_pkt.columns:
    if "label" in df_pkt.columns:
        df_pkt["label_bin"] = df_pkt["label"].astype(int)
    else:
        raise ValueError("업로드한 파일에 label 컬럼이 없습니다!")

# 4) 두 데이터의 컬럼 구조 맞추기
missing_cols = set(df_main.columns) - set(df_pkt.columns)
for col in missing_cols:
    df_pkt[col] = None  # 없는 컬럼은 None으로 채움

# df_pkt 컬럼 순서를 df_main과 동일하게
df_pkt = df_pkt[df_main.columns]

# 5) 병합
df_merged = pd.concat([df_main, df_pkt], axis=0, ignore_index=True)

# 6) label_bin NaN 제거 (패킷 데이터 외 데이터는 정상)
df_merged = df_merged[df_merged["label_bin"].notna()]

# 최종 저장
df_merged.to_csv(OUTPUT_PATH, sep="\t", index=False)
print("정상적으로 병합 및 label_bin 정리 완료:", OUTPUT_PATH)

import pandas as pd
from sklearn.model_selection import train_test_split

# ================================
# 1) 경로 설정
# ================================
INPUT_PATH = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/merged_final_clean.tsv"
OUTPUT_DIR = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/splits/"

!mkdir -p $OUTPUT_DIR

# ================================
# 2) TSV 로드
# ================================
print("[INFO] TSV 로드 중...")
df = pd.read_csv(INPUT_PATH, sep="\t", low_memory=False)

print("전체 데이터 크기:", df.shape)

# label_bin NaN 제거
df = df[df["label_bin"].notna()]
df["label_bin"] = df["label_bin"].astype(int)

print("label_bin NaN 제거 후:", df.shape)


# ================================
# 3) train / val / test 분할
# ================================
print("[INFO] train/val/test 분할 중...")

# train: 80%, val: 10%, test: 10%
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label_bin"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label_bin"],
    random_state=42
)

print(f"train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")


# ================================
# 4) TSV로 저장
# ================================
train_path = OUTPUT_DIR + "train.tsv"
val_path = OUTPUT_DIR + "val.tsv"
test_path = OUTPUT_DIR + "test.tsv"

train_df.to_csv(train_path, sep="\t", index=False)
val_df.to_csv(val_path, sep="\t", index=False)
test_df.to_csv(test_path, sep="\t", index=False)

print("\n[완료] 데이터 저장 위치:")
print("Train:", train_path)
print("Val  :", val_path)
print("Test :", test_path)


import pandas as pd

split_dir = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/splits"

for split in ["train", "val", "test"]:
    path = f"{split_dir}/{split}.tsv"
    print("처리 중:", path)

    # TSV 로드
    df = pd.read_csv(path, sep="\t", dtype=str)

    # NaN 라벨 행 제거
    df = df.dropna(subset=["label"])

    # float → int 변환
    df["label"] = df["label"].astype(float).astype(int)

    # 0은 그대로, 나머지는 1로 변환
    df["label"] = df["label"].apply(lambda x: 0 if x == 0 else 1)

    # 저장
    df.to_csv(path, sep="\t", index=False)

    # 확인
    print(split, "split 완료. 0/1 개수:", df["label"].value_counts().to_dict())

print("모든 split 파일에서 NaN 제거 및 binary label 변환 완료!")




