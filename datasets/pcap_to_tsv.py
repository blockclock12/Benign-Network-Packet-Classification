import subprocess
import csv
import sys
import os
'''
pcap파일을 tsv파일로 변환시키는 코드, 73번째 줄의 desired_fields의 리스트 요소를 바꾸면 필요한 정보를 선택할 수 있음
'''

def pcap_to_tsv_tshark(input_pcap, output_tsv, fields):
    """
    pcap 파일에서 지정된 필드만 추출하여 TSV 파일로 변환합니다.

    :param input_pcap: 입력 .pcap 파일 경로
    :param output_tsv: 출력 .tsv 파일 경로
    :param fields: 추출할 필드 이름 목록 (예: ['frame.number', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport'])
    """
    # TShark 명령 구성
    # -r: 입력 파일 지정
    # -T fields: 출력 형식을 필드 기반으로 지정
    # -E separator=\t: 필드 구분자를 탭(\t)으로 지정 (TSV)
    # -E header=y: 첫 줄에 필드 이름(헤더) 포함
    # -e: 추출할 필드 지정 (리스트를 문자열로 변환하여 사용)
    
    field_args = [arg for field in fields for arg in ('-e', field)]
    
    tshark_command = [
        'tshark',
        '-r', input_pcap,
        '-T', 'fields',
        '-E', 'separator=\t',
        '-E', 'header=y',
        *field_args # 추출할 필드 리스트
    ]
    
    print(f"TShark 명령: {' '.join(tshark_command)}")
    print(f"'{input_pcap}' 파일을 처리 중입니다...")

    try:
        # TShark 실행 및 결과 캡처
        # stderr는 출력으로 오지 않도록 PIPE로 설정
        process = subprocess.run(
            tshark_command,
            capture_output=True,
            text=True,
            check=True,  # 오류 발생 시 예외 발생
            encoding='utf-8'
        )
        
        # 결과를 TSV 파일로 저장
        with open(output_tsv, 'w', newline='', encoding='utf-8') as outfile:
            outfile.write(process.stdout)
            
        print(f"성공적으로 '{output_tsv}' 파일에 저장되었습니다.")
        
    except FileNotFoundError:
        print(f"오류: 'tshark' 명령을 찾을 수 없습니다. Wireshark/TShark가 설치되어 있고 PATH에 등록되었는지 확인하십시오.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"TShark 실행 중 오류 발생:", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}", file=sys.stderr)

# --- 사용 예시 ---

# 1. 파일 경로 설정
input_file = 'test.pcap' # 실제 PCAP 파일 이름으로 변경하세요
output_file = 'extracted_data.tsv'

# 2. 추출을 원하는 필드 목록 설정
# Wireshark/TShark 필터 레퍼런스(예: ip.src, tcp.dstport, http.request.uri 등)를 참조하세요.
# 참고: _ws.col.Info 및 _ws.col.Protocol은 Wireshark의 'Summary' 정보 필드입니다.
desired_fields = [
    #'frame.number', 
    #'frame.time',
    'ip.src', 
    'ip.dst', 
    '_ws.col.Protocol', 
    #'tcp.srcport', 
    #'tcp.dstport',
    #'_ws.col.Info',
    'tcp.payload'
]

# (테스트용) 샘플 PCAP 파일이 존재하는지 확인
if not os.path.exists(input_file):
    print(f"경고: 입력 파일 '{input_file}'을 찾을 수 없습니다. 테스트를 위해 파일 이름을 변경하거나 파일을 준비해주세요.", file=sys.stderr)
else:
    pcap_to_tsv_tshark(input_file, output_file, desired_fields)
