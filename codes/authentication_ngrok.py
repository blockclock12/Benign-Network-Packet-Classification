import os
import subprocess
import time
from pyngrok import ngrok, conf

# (★필수★) ngrok 인증 토큰 설정
AUTH_TOKEN = "35MpYV1cFwjpXAE43NPdUp3iuCU_5CMBn6g4ynGF8m6vk1aGm"

if AUTH_TOKEN == "YOUR_NGROK_AUTHTOKEN_HERE":
    print("="*50)
    print("오류: ngrok 인증 토큰(AUTH_TOKEN)을 설정해주세요.")
    print("https://dashboard.ngrok.com/get-started/your-authtoken")
    print("="*50)
else:
    conf.get_default().auth_token = AUTH_TOKEN

    # Streamlit을 백그라운드에서 실행 (포트 8501)
    try:
        process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
        print("--- Streamlit 서버 시작 중... (5초 대기) ---")
        time.sleep(5)

        # ngrok 터널 연결 (포트 8501)
        public_url = ngrok.connect(8501)
        print(f"============================================================")
        print(f"Streamlit 앱이 다음 공개 URL에서 실행 중입니다:")
        print(f"{public_url}")
        print(f"============================================================")

    except Exception as e:
        print(f"서버 실행/ngrok 연결 오류: {e}")
