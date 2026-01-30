# ---------------------------
# 1) NSL-KDD
NSL_DIR = '/content/NSL-KDD-Dataset'
if not os.path.exists(NSL_DIR):
    !git clone https://github.com/jmnwong/NSL-KDD-Dataset.git {NSL_DIR}

def load_nsl():
    nsl_df = pd.DataFrame()
    try:
        f_train = os.path.join(NSL_DIR, 'KDDTrain+.txt')
        f_test  = os.path.join(NSL_DIR, 'KDDTest+.txt')
        if os.path.exists(f_train):
            df_t = pd.read_csv(f_train, header=None)
            df_e = pd.read_csv(f_test, header=None) if os.path.exists(f_test) else pd.DataFrame()
            nsl_df = pd.concat([df_t, df_e], ignore_index=True)
            cols = [f'feature_{i}' for i in range(nsl_df.shape[1]-1)] + ['label']
            nsl_df.columns = cols
            nsl_df['source'] = 'NSL-KDD'
    except:
        pass
    print('NSL-KDD 로드 완료, 행수:', len(nsl_df))
    return nsl_df

# ---------------------------
# 2) CICIDS2018
CICIDS_DIR = '/content/CICIDS2018'
if not os.path.exists(CICIDS_DIR):
    !git clone https://github.com/IDS-AI/CICIDS2018.git {CICIDS_DIR}

def load_cicids():
    files = glob.glob(os.path.join(CICIDS_DIR, '**', '*.csv'), recursive=True)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source'] = 'CICIDS2018'
            dfs.append(df)
        except:
            continue
    if dfs:
        out = pd.concat(dfs, ignore_index=True, sort=False)
        print('CICIDS2018 로드 완료, 행수:', len(out))
        return out
    return pd.DataFrame()

# ---------------------------
# 3) CTU-13
CTU_DIR = '/content/CTU13-CSV-Dataset'
if not os.path.exists(CTU_DIR):
    !git clone https://github.com/imfaisalmalik/CTU13-CSV-Dataset.git {CTU_DIR}

def load_ctu13():
    files = glob.glob(os.path.join(CTU_DIR, '**', '*.csv'), recursive=True)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source'] = 'CTU-13'
            dfs.append(df)
        except:
            continue
    if dfs:
        out = pd.concat(dfs, ignore_index=True, sort=False)
        print('CTU-13 로드 완료, 행수:', len(out))
        return out
    print('CTU-13 로드: 사용 가능한 CSV 없음')
    return pd.DataFrame()
