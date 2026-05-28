from fastapi import FastAPI, Query
import requests

app = FastAPI()

# 페이스북 MusicGen 정식 AI가 돌아가고 있는 허깅페이스 무료 API 주소
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
# 💡 만약 나중에 공식 토큰이 필요하면 헤더에 추가할 수 있도록 세팅
HEADERS = {} 

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
    print(f"⚙️ [옵션] 보컬제거: {remove_vocal} / 주멜로디뮤트: {mute_melody}")

    # 허깅페이스 AI에게 진수님이 앱에서 입력한 프롬프트 전달하기
    payload = {"inputs": prompt}
    
    try:
        print("🚀 [AI 연성 중] 허깅페이스 고성능 컴퓨터로 프롬프트 전송...")
        response = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=60)
        
        if response.status_code == 200:
            print("✅ [연성 완료] AI 음악 바이너리 데이터 수신 성공!")
            # 나중에 여기에 Render 서버나 Storage에 MP3 파일로 저장하고 다운로드 링크를 넘겨주는 로컬 코드가 얹어집니다.
            return {
                "status": "success",
                "message": "AI 음악 생성이 완료되었습니다.",
                "user_prompt": prompt,
                "preview_info": f"HuggingFace 전송 성공 (Size: {len(response.content)} bytes)"
            }
        else:
            print(f"❌ [AI 에러] 허깅페이스 응답 실패: {response.status_code}")
            return {"status": "error", "message": f"AI 엔진 응답 실패 (코드: {response.status_code})"}
            
    except Exception as e:
        print(f"💥 [서버 에러] 통신 중 오류 발생: {str(e)}")
        return {"status": "error", "message": str(e)}
