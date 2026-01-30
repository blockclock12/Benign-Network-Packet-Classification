%%writefile app.py
import streamlit as st
import subprocess
import csv
import sys
import os
import pandas as pd
import json
import argparse
import torch
import torch.nn as nn

# --- (★수정★) 페이지 레이아웃 설정 ---
# 'wide' 레이아웃으로 설정하고 페이지 제목(브라우저 탭)을 지정합니다.
# 이 코드는 항상 st.xxx 명령어 중 가장 먼저 실행되어야 합니다.
st.set_page_config(layout="wide", page_title="악성 트래픽 분석 대시보드")

# --- (★추가★) 사이드바 UI ---
with st.sidebar:
    st.title("ET-BERT")
    st.write("---")
    # (아이콘은 streamlit이 지원하는 이모지로 대체합니다)
    # 링크를 '#'로 설정하여 현재 페이지에 머무르도록 합니다.
    st.link_button("대시보드", "#", icon="🏠", use_container_width=True)
    st.link_button("경고", "#", icon="⚠️", use_container_width=True)
    st.link_button("설정", "#", icon="⚙️", use_container_width=True)
# --- 사이드바 끝 ---

# --- (A) ET-BERT 모델 로드 로직 (main.py에서 가져옴) ---
# 이 로직은 ET-BERT 모델을 로드하는 데 필요합니다.

# 1. "uer" 부품 로드 로직
repo_path = "/content/drive/MyDrive/ET-BERT-main/" # Colab 경로 기준
if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from uer.layers import *
    from uer.encoders import *
    from uer.utils import *
    from uer.utils.vocab import Vocab
    from uer.utils.constants import *
except ImportError as e:
    st.error(f"Import 오류: {e}. 'uer' 폴더 경로({repo_path})를 찾을 수 없습니다.")
    st.stop()

# 2. "모델 뼈대" (Classifier) 정의
class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, args.vocab_size)
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1,]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits

# 3. 모델/토크나이저/설정값을 로드하는 함수
# @st.cache_resource: 이 함수는 한 번만 실행되고 결과가 캐시됩니다.
@st.cache_resource
def load_all_resources():
    DRIVE_PATH = "/content/drive/MyDrive/ET-BERT-main/"
    MODEL_PATH = os.path.join(DRIVE_PATH, "fine-tuning/USTC-TFC_results/finetuned_model.bin")
    VOCAB_PATH = os.path.join(DRIVE_PATH, "models/encryptd_vocab.txt")
    CONFIG_PATH = os.path.join(DRIVE_PATH, "models/bert_base_config.json")

    try:
        args = argparse.Namespace()
        with open(CONFIG_PATH, "r") as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(args, key, value)

        args.labels_num = 2
        vocab_size = 0
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                vocab_size += 1
        args.vocab_size = vocab_size

        # --- 모든 수동 설정값 ---
        args.pooling = "first"
        args.soft_targets = False
        args.soft_alpha = 0.5
        args.tokenizer = "bert"
        args.encoder = "transformer"
        args.mask = "fully_visible"
        args.embedding = "word_pos_seg"
        args.remove_embedding_layernorm = False
        args.parameter_sharing = False
        args.factorized_embedding_parameterization = False
        args.layernorm_positioning = "pre"
        args.remove_transformer_bias = False
        args.remove_attention_scale = False
        args.has_residual_attention = False
        args.relative_position_embedding = False
        args.feed_forward = "linear"
        args.layernorm = "normal"

        # --- 이름 매핑 ---
        if hasattr(args, 'max_position_embeddings'):
            args.max_seq_length = args.max_position_embeddings
        else:
            args.max_seq_length = 512
        if hasattr(args, 'intermediate_size') and not hasattr(args, 'feedforward_size'):
            args.feedforward_size = args.intermediate_size
        if hasattr(args, 'num_attention_heads') and not hasattr(args, 'heads_num'):
            args.heads_num = args.num_attention_heads
        if hasattr(args, 'hidden_dropout_prob') and not hasattr(args, 'dropout'):
            args.dropout = args.hidden_dropout_prob
        if hasattr(args, 'hidden_size') and not hasattr(args, 'emb_size'):
            args.emb_size = args.hidden_size

        args.vocab_path = VOCAB_PATH
        args.spm_model_path = None
        args.config_path = CONFIG_PATH

        tokenizer = str2tokenizer[args.tokenizer](args)
        model = Classifier(args)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')), strict=False)
        model.eval()

        print("--- 🥳 모델/토크나이저 로드 성공 ---")
        return model, tokenizer, args

    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

# --- (B) 사용자님의 PCAP 변환 함수 ---
def pcap_to_tsv_tshark(input_pcap, output_tsv, fields):
    field_args = [arg for field in fields for arg in ('-e', field)]
    tshark_command = [
        'tshark',
        '-r', input_pcap,
        '-T', 'fields',
        '-E', 'separator=\t',
        '-E', 'header=y',
        *field_args
    ]

    try:
        process = subprocess.run(
            tshark_command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        with open(output_tsv, 'w', newline='', encoding='utf-8') as outfile:
            outfile.write(process.stdout)
        return True, None # 성공

    except FileNotFoundError:
        return False, "오류: 'tshark' 명령을 찾을 수 없습니다. (Colab에서 !apt-get install -y tshark 를 실행했는지 확인하세요)"
    except subprocess.CalledProcessError as e:
        return False, f"TShark 오류: {e.stderr}"
    except Exception as e:
        return False, f"예기치 않은 오류: {e}"

# --- (C) ET-BERT 모델 추론 함수 ---
def predict_hex_string(model, tokenizer, args, hex_string):
    seq_length = 128 # 학습 시 사용한 seq_length
    try:
        hex_string_spaced = " ".join([hex_string[i:i+2] for i in range(0, len(hex_string), 2)])
        src_ids = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(hex_string_spaced))
        seg_ids = [1] * len(src_ids)

        if len(src_ids) > seq_length:
            src_ids = src_ids[:seq_length]
            seg_ids = seg_ids[:seq_length]
        while len(src_ids) < seq_length:
            src_ids.append(0)
            seg_ids.append(0)

        input_tensor = torch.LongTensor([src_ids])
        segment_tensor = torch.LongTensor([seg_ids])

        with torch.no_grad():
            logits = model(input_tensor, segment_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        label_map = {0: "Benign", 1: "Malicious"}
        label = label_map.get(predicted_class.item(), "Unknown")

        return label, confidence.item()

    except Exception as e:
        st.error(f"모델 추론 오류: {e}")
        return "Error", 0.0

# --- (D) Streamlit UI 메인 로직 ---

# (★수정★) 제목 및 상태 배지
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("악성 트래픽 분석 대시보드")
    st.caption("모니터링 및 분석")
# with col_status:
    # (임시) 상태 배지
    # st.error("상태: ① 위험 상태", icon="🔥")

# 1. 모델/토크나이저 로드 (캐시됨)
try:
    model, tokenizer, args = load_all_resources()
    st.success("모델과 토크나이저 로드 완료!")
except Exception as e:
    # load_all_resources 내부에서 이미 st.error를 호출하지만, 만약을 위해 이중 체크
    st.error(f"모델 로드 중 심각한 오류 발생: {e}")
    st.stop() # 모델 로드 실패 시 앱 중지

# 2. 파일 업로드 UI (대시보드 레이아웃 중앙에 배치)
st.write("---")
# (★수정★) 파일 업로더를 중앙 컬럼에 배치하여 스크린샷과 유사하게 만듭니다.
col1_up, col2_up, col3_up = st.columns([1, 2, 1])
with col2_up:
    uploaded_file = st.file_uploader(
        "PCAP 파일을 여기에 드래그 앤 드롭하세요.",
        type=["pcap", "pcapng"],
        label_visibility="hidden" # "Drag and drop" 텍스트가 기본이므로 라벨 숨김
    )
st.write("---")

# 3. (★수정★) 결과 표시: 3단 컬럼 레이아웃
if uploaded_file is not None:

    # 3-1. pcap 변환
    INPUT_PCAP = "temp_uploaded.pcap"
    OUTPUT_TSV = "temp_output.tsv"
    with open(INPUT_PCAP, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner(f"{uploaded_file.name} 파일 변환 중... (tshark 실행)"):
        # (★수정★) IP 주소도 함께 추출합니다.
        desired_fields = ['ip.src', 'ip.dst', 'tcp.payload']
        success, error_msg = pcap_to_tsv_tshark(INPUT_PCAP, OUTPUT_TSV, desired_fields)

    if not success:
        st.error(error_msg)
    else:
        # 3-2. TSV 파일 읽기
        try:
            df = pd.read_csv(OUTPUT_TSV, sep='\t')
            if df.empty:
                 st.error("pcap 파일에서 분석할 수 있는 패킷(TCP Payload)을 찾지 못했습니다.")
                 st.stop()

            # 16진수 및 IP 정보 추출
            hex_string = str(df.iloc[0].get('tcp.payload', 'N/A')).replace(":", "")
            ip_src = str(df.iloc[0].get('ip.src', 'N/A'))
            ip_dst = str(df.iloc[0].get('ip.dst', 'N/A'))

            if hex_string == 'N/A':
                st.error("'tcp.payload' 필드를 TSV에서 찾을 수 없습니다.")
                st.stop()

            # 3-3. 모델 추론
            with st.spinner("ET-BERT 모델이 추론 중..."):
                label, confidence = predict_hex_string(model, tokenizer, args, hex_string)

            # 3-4. (★수정★) 3단 컬럼으로 결과 표시
            col1, col2, col3 = st.columns(3)

            # --- 카드 1: 전체 위험 수준 ---
            with col1:
                st.subheader("전체 위험 수준")
                if "Malicious" in label:
                    st.metric("탐지 결과", label, "심각", delta_color="inverse")
                    st.caption("범례: 🔴 심각 🟡 주의 🟢 안전")
                else:
                    st.metric("탐지 결과", label, "안전", delta_color="normal")
                    st.caption("범례: 🔴 심각 🟡 주의 🟢 안전")

                st.write("신뢰도 (게이지):")
                st.progress(confidence)


            # --- 카드 2: 주요 탐지 위협 ---
            with col2:
                st.subheader("주요 탐지 위협 (예상)")
                if "Malicious" in label:
                    # (이 데이터는 ET-BERT가 반환하지 않으므로 *가짜* 데이터입니다)
                    threat_data = {
                        "Threat": ["Trojan.Zeus", "C&C 서버", "Botnet"],
                        "Percentage": [confidence * 0.85, confidence * 0.65, confidence * 0.48]
                    }
                    # Create DataFrame, set index for better bar chart labels
                    df_threat = pd.DataFrame(threat_data).set_index("Threat")
                    st.bar_chart(df_threat, y="Percentage") # Use y= to match screenshot's vertical bars
                else:
                    st.info("탐지된 주요 위협 없음.")

            # --- 카드 3: 네트워크 상태 ---
            with col3:
                st.subheader("네트워크 정보")
                st.text(f"출발지 IP: {ip_src}")
                st.text(f"목적지 IP: {ip_dst}")
                st.write("---")
                st.metric("모델 신뢰도", f"{confidence*100:.1f}%")

        except pd.errors.EmptyDataError:
            st.error("생성된 TSV 파일이 비어있습니다. pcap 파일 내용을 확인하세요.")
        except Exception as e:
            st.error(f"TSV 파일 처리 중 오류: {e}")
            import traceback
            st.error(traceback.format_exc())
