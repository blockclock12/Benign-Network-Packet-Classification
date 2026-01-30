import os
import sys
import glob
import shutil
import time
import subprocess
import pandas as pd

# 1) 필요한 도구 설치 (aws cli)
print('\\n==> 1) aws cli 설치(혹시 이미 있으면 무시됩니다)')
!pip install -q awscli

# AWS S3에서 CICIDS2018 CSV 일부만 가져오기
CICIDS18_LOCAL = '/content/CICIDS2018'
os.makedirs(CICIDS18_LOCAL, exist_ok=True)
S3_PREFIX = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

try:
    res = subprocess.run(['aws','s3','ls', S3_PREFIX, '--no-sign-request'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if res.returncode == 0:
        print('S3 경로 접근 성공 — CSV 파일 일부만 복사')
        # 가져올 파일 목록
        include_files = [
            'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
            'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
            'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
        ]
        for pattern in include_files:
            cmd = ['aws','s3','cp', S3_PREFIX, CICIDS18_LOCAL,
                   '--recursive','--no-sign-request','--exclude','*','--include',pattern]
            print('실행:', ' '.join(cmd))
            r2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
            print(r2.stdout)
            if r2.returncode != 0:
                print(f'{pattern} 복사 실패 (stderr): {r2.stderr}')
    else:
        print('S3 접근 실패 (public S3가 아닐 수 있음). 출력:')
        print(res.stderr)
except Exception as e:
    print('CIC-IDS2018 S3 접근 오류:', e)


# 3) CSE-CIC-IDS2018: AWS S3에서 CSV 폴더 복사 시도 (no-sign-request)
print('\n==> 3) CSE-CIC-IDS2018 (AWS S3)에서 CSV 복사 시도')
CICIDS18_LOCAL = '/content/CICIDS2018'
os.makedirs(CICIDS18_LOCAL, exist_ok=True)
S3_PREFIX = "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"

try:
    res = subprocess.run(['aws','s3','ls', S3_PREFIX, '--no-sign-request'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    if res.returncode == 0:
        print('S3 경로 접근 성공 — CSV 파일을 복사합니다 (용량에 유의).')
        cmd = ['aws','s3','cp', S3_PREFIX, CICIDS18_LOCAL, '--recursive', '--no-sign-request', '--exclude', '*', '--include', '*TrafficForML*.csv']
        print('실행:', ' '.join(cmd))
        r2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1200)
        print(r2.stdout)
        if r2.returncode != 0:
            print('CIC-IDS2018 복사 실패 또는 파일 없음 (stdout/stderr):')
            print(r2.stderr)
    else:
        print('S3 접근 실패 (public S3가 아닐 수 있습니다). 출력:')
        print(res.stderr)
except Exception as e:
    print('CIC-IDS2018 S3 접근 오류:', e)



# ---------------------------
# 4) 데이터 로드 및 통합
nsl_df   = load_nsl()
cicids_df = load_cicids()
ctu_df   = load_ctu13()

dfs = [df for df in [nsl_df, cicids_df, ctu_df] if not df.empty]
if not dfs:
    raise SystemExit('모든 데이터 로드 실패! (각 저장소의 CSV 존재 여부 확인)')

combined = pd.concat(dfs, ignore_index=True, sort=False)
print('통합 전 총 행수:', len(combined))


# ---------------------------
# 5) 라벨 정규화
def normalize_label(s):
    if pd.isna(s):
        return 'malicious'
    s = str(s).lower()
    if any(k in s for k in ['normal','benign','background','0']):
        return 'benign'
    if any(k in s for k in ['attack','malicious','dos','ddos','bot','scan']):
        return 'malicious'
    return 'malicious'

if 'label' in combined.columns:
    combined['label_raw'] = combined['label']
else:
    combined['label_raw'] = combined.iloc[:, -1].astype(str)

combined['norm_label'] = combined['label_raw'].apply(normalize_label)
combined['label_bin'] = combined['norm_label'].apply(lambda x: 0 if x=='benign' else 1)



# ---------------------------
# 6) ET-BERT 학습용 text 생성
def row_to_text(row):
    parts = []
    skip = set(['label','label_raw','norm_label','label_bin','source'])
    cnt = 0
    for c in row.index:
        if c in skip:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        parts.append(f'{c}={str(v)[:80]}')
        cnt += 1
        if cnt >= 12:
            break
    return ' | '.join(parts)

combined['text'] = combined.apply(row_to_text, axis=1)
combined = combined.dropna(subset=['text']).reset_index(drop=True)


# ==================================================
# 6) 저장 경로 설정
BASE_DIR = '/content/drive/MyDrive/etbert_datasets'
OUT_DIR = os.path.join(BASE_DIR, 'merged_etbert_dataset_v4')
os.makedirs(OUT_DIR, exist_ok=True)

csv_path = os.path.join(OUT_DIR, 'merged_etbert_dataset.csv')
parquet_path = os.path.join(OUT_DIR, 'merged_etbert_dataset.parquet')


# ==================================================
# 7) object 컬럼에 int/float 섞인 경우 대비, 모두 문자열로 변환
for c in combined.columns:
    if combined[c].dtype == 'object':
        combined[c] = combined[c].astype(str)


# ==================================================
# 8) 저장
combined.to_csv(csv_path, index=False)
combined.to_parquet(parquet_path, index=False)

print(' 통합 데이터 저장 완료')
print(' CSV:', csv_path)
print(' PARQUET:', parquet_path)
print(' 샘플 수:', len(combined))



# ==================================================
# 9) tsv 변환
import pandas as pd

csv_path = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/merged_etbert_dataset.csv"
tsv_path = "/content/drive/MyDrive/etbert_datasets/merged_etbert_dataset_v4/merged_etbert_dataset.tsv"

df = pd.read_csv(csv_path)

# TSV로 저장
df.to_csv(tsv_path, sep='\t', index=False)

print("TSV 저장 완료:", tsv_path)
