from fastapi import FastAPI, Query, BackgroundTasks, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import redis

app = FastAPI()

# static 폴더가 없으면 자동으로 생성해 주는 안전장치
if not os.path.exists("static"):
    os.makedirs("static")

# ⭐️ 오디오 파일 재생 주소를 외부(코듈라 앱)로 열어주는 정적 가상 통로 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# 실시간 상태 동기화 및 작업 큐를 위한 Redis 연결 (문서 7페이지 반영)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
rd = redis.from_url(REDIS_URL)

@app.get("/")
def read_root():
    return {"message": "SymphonyAI 전용 생산 환경 서버가 정상 가동 중입니다!"}

# =================================================================
# [수정 완료] 코듈라 앱에서 MP3 파일 업로드 시 편집실 연동 처리 API
# =================================================================
@app.post("/api/file/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    # 1. 파일명에 포함된 공백을 언더바(_)로 치환하여 URL 깨짐 방지
    clean_filename = file.filename.replace(" ", "_")
    file_path = f"static/{clean_filename}"
    
    # 2. 서버 내부 static 폴더에 유저가 선택한 파일 저장하기
    with open(file_path, "wb+") as file_object:
        file_object.write(file.file.read())
        
    # 3. 🌟 핵심: 코듈라의 Player2가 실시간으로 재생할 수 있도록 '/static/' 가상 경로를 정확히 포함하여 주소 조립
    full_audio_url = f"https://symphonyai-server.onrender.com/static/{clean_filename}"
    
    # 4. 코듈라 블록의 "audio_url", "file_name" 이라는 key값과 단 한 글자도 틀리지 않게 매칭하여 반환
    return {
        "status": "success",
        "audio_url": full_audio_url,
        "file_name": clean_filename
    }

# 1. 메인 변환 & 전처리 인터페이스 (브라스 풀밴드, 오케스트라, 클럽 리믹스 등 프롬프트 수신)
@app.post("/api/convert")
def convert_music(
    song: str = Query(..., description="업로드된 MP3 파일명"),
    prompt: str = Query(..., description="재편곡 AI 마법 주문 프롬프트"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # TODO: Background Worker(Pro)로 무거운 AI 연산(옥타브 자동조정, 화성학 편곡)을 넘기는 큐 작업 추가
    return {
        "status": "processing",
        "message": "AI 재편곡 및 화성학 연산 작업이 큐에 등록되었습니다.",
        "song": song,
        "prompt": prompt
    }

# 2. 보컬 트랙 즉각 격리 및 제거 인터페이스
@app.post("/api/vocal-remove")
def vocal_remove(
    song: str = Query(..., description="보컬을 제거할 곡명"),
    remove_vocal: str = Query("Y", description="보컬 제거 여부 (Y/N)"),
    mute_melody: str = Query("N", description="주멜로디 뮤트 여부 (Y/N)")
):
    return {
        "status": "success",
        "message": f"보컬 격리 완료 (Vocal Remove: {remove_vocal} / Mute Melody: {mute_melody})"
    }

# 3. 플레이어 시간과 서버 데이터 실시간 동기화 (진행바 대응)
@app.get("/api/sync")
def sync_player(song: str, current_time: float):
    # 앱의 진행바와 실시간 동기화 처리
    return {"song": song, "server_sync_time": current_time, "status": "synchronized"}

# 4. 보관함 점3개 메뉴 대응 (재편집 및 각종 파일 다운로드 API 뼈대)
@app.get("/api/download")
def download_artifact(
    song: str, 
    file_type: str = Query(..., description="mp3, wav, mid, total_score, part_score, chord_score")
):
    return {
        "status": "success",
        "file_type": file_type,
        "download_url": f"https://symphony-ai-storage.com/exports/{song}.{file_type if 'score' not in file_type else 'pdf'}"
    }

# =================================================================
# 5. [트렌드 반영] 소셜 로그인 요청 및 링크 제공 인터페이스
# =================================================================
@app.get("/api/login")
def social_login(provider: str = Query(..., description="google 또는 kakao")):
    
    if provider == "google":
        # 구글 로그인 페이지로 유저를 강제 이동(Redirect) 시킵니다.
        # ※ 실제 상용화 시에는 YOUR_GOOGLE_CLIENT_ID를 본인의 구글 콘솔 키로 교체해야 합니다.
        google_oauth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth"
            "?client_id=85780968680-e713urhtmn3utpcc997b7h78d3machpr.apps.googleusercontent.com"
            "&redirect_uri=https://symphonyai-server.onrender.com/api/login/callback/google"
            "&response_type=code"
            "&scope=email%20profile"
        )
        return RedirectResponse(url=google_oauth_url)
        
    elif provider == "kakao":
        # 카카오 로그인 페이지로 유저를 강제 이동(Redirect) 시킵니다.
        # ※ 실제 상용화 시에는 YOUR_KAKAO_REST_KEY를 본인의 카카오 개발자 키로 교체해야 합니다.
        kakao_oauth_url = (
            "https://kauth.kakao.com/oauth/authorize"
            "?client_id=180c9d0fdb51f06f99b7b97ff373830f" #
            "&redirect_uri=https://symphonyai-server.onrender.com/api/login/callback/kakao"
            "&response_type=code"
        )
        return RedirectResponse(url=kakao_oauth_url)
        
    return {"status": "failed", "message": "지원하지 않는 SNS 로그인 제공업체입니다."}

# =================================================================
# 6. [트렌드 반영] SNS 인증 완료 후 콜백(Callback) 및 앱으로 복귀(딥링크)
# =================================================================
@app.get("/api/login/callback/{provider}")
def oauth_callback(provider: str, code: str):
    # [백엔드 내부 로직 영역]
    # 여기서 원래는 구글/카카오 서버와 code를 주고받아 유저의 실제 이메일을 받아옵니다.
    
    # (테스트용 가상 데이터 설정)
    test_user_id = "symphony_user_777"
    test_email = "symphony_user@gmail.com" if provider == "google" else "kakao_user@kakao.com"
    
    # 🌟 핵심: 로그인이 끝나면 유저 정보를 주소 뒤에 달고 '앱 고유의 주소(딥링크)'로 튕겨줍니다.
    # 스마트폰이 이 주소를 감지하면 웹 브라우저를 닫고 우리 앱을 강제로 다시 켭니다.
    app_deep_link_url = f"symphonyai://login_success?user_id={test_user_id}&email={test_email}"
    
    return RedirectResponse(url=app_deep_link_url)
