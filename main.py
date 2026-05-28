from fastapi import FastAPI, Query
import requests
import os  # 👈 1. 이 줄이 새로 추가되었습니다!

app = FastAPI()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/pipeline/text-to-audio/facebook/musicgen-small"

# 👈 2. 기존 HEADERS = {} 부분을 아래 내용으로 통째로 교체합니다!
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

@app.get("/")
def read_root():
    return {"message": "SymphonyAI 무료 가속 서버가 정상 작동 중입니다!"}

@app.get("/api/convert")
def convert_music(
    song: str = Query(..., description="변환할 곡 파일명"),
    prompt: str = Query(..., description="AI 마법 주문 프롬프트"),
    remove_vocal: str = Query("N", description="보컬 제거 여부 (Y/N)"),
    mute_melody: str = Query("N", description="주멜로디 뮤트 여부 (Y/N)")
):
    print(f"🎵 [주문 접수] 곡명: {song}")
    print(f"✍️ [마법 주문] 프롬프트: {prompt}")
    
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=60)
        
        if response.status_code == 200:
            return {
                "status": "success",
                "message": "AI 음악 생성이 완료되었습니다.",
                "user_prompt": prompt,
                "preview_info": f"HuggingFace 인증 통신 성공 (Size: {len(response.content)} bytes)"
            }
        else:
            return {"status": "error", "message": f"AI 엔진 응답 실패 (코드: {response.status_code})"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}
