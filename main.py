from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SymphonyAI 메인 음악 생성 서버")

# 🌐 앱에서 서버로 접근할 수 있도록 보안(CORS) 문을 활짝 열어줍니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "running", "message": "SymphonyAI 전용 서버가 정상 작동 중입니다."}

# 🔥 진수님이 앱에서 5단 콤보로 쏘아 올린 요청을 받아내는 '진짜 접수창구'
@app.get("/api/convert")
def generate_music(
    song: str = Query(..., description="선택된 오디오 파일명"),
    prompt: str = Query("", description="유저가 입력한 마법 주문 (예: 브라스 풀밴드)"),
    remove_vocal: str = Query("N", description="보컬 제거 여부"),
    mute_melody: str = Query("N", description="주멜로디 뮤트 여부")
):
    # 앱에서 신호가 오면 컴퓨터 검은 창(터미널)에 아래 로그가 똑똑히 찍힙니다.
    print(f"🎵 [주문 접수] 곡명: {song}")
    print(f"✍️ [마법 주문] 프롬프트: {prompt}")
    print(f"🎤 [옵션] 보컬제거: {remove_vocal} / 주멜로디뮤트: {mute_melody}")
    
    # [여기에 Meta MusicGen 무료 AI 엔진이 들어가서 뚝딱뚱땅 음악을 연성할 예정!]
    
    return {
        "status": "success",
        "user_prompt": prompt,
        "message": f"'{prompt}' 스타일로 오디오 변환 작업을 시작합니다!"
    }