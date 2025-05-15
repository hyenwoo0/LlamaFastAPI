# OS 및 로깅 관련 기본 모듈 임포트
import os
import logging
import time

# FastAPI 관련 모듈
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # CORS 허용 설정
from fastapi.responses import JSONResponse          # JSON 응답 반환

# 요청 본문 유효성 검사용 Pydantic
from pydantic import BaseModel

# 속도 제한 라이브러리 SlowAPI
from slowapi import Limiter
from slowapi.util import get_remote_address         # 클라이언트 IP 가져오기
from slowapi.errors import RateLimitExceeded        # 속도 초과 예외처리용

# LLaMA 모델 로딩을 위한 llama-cpp 파이썬 바인딩
from llama_cpp import Llama

# Jupyter나 PyCharm 환경에서도 asyncio 충돌 방지용
import nest_asyncio

# FastAPI 실행 서버
import uvicorn

# ------------------------- 로그 설정 -------------------------
LOG_DIR = "./logs"                                  # 로그 저장 폴더
os.makedirs(LOG_DIR, exist_ok=True)                 # 폴더 없으면 생성
log_file_path = os.path.join(LOG_DIR, "chat_server.log")  # 로그 파일 경로 설정

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,                             # 로그 레벨: INFO
    format="%(asctime)s [%(levelname)s] %(message)s",  # 로그 메시지 포맷
    handlers=[
        logging.StreamHandler(),                    # 콘솔 출력 핸들러
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 파일 저장 핸들러
    ]
)

# ------------------------- LLaMA 모델 로딩 -------------------------
MODEL_PATH = "/Users/johyeon-u/PycharmProjects/LlamaFastAPI/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf"  # 모델 경로
logging.info("🔄 모델 로딩 중...")                    # 모델 로딩 시작 로그
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)       # llama.cpp 모델 로딩, context window 설정
logging.info("✅ 모델 로딩 완료")                    # 모델 로딩 완료 로그

# ------------------------- FastAPI 초기화 및 속도 제한 -------------------------
app = FastAPI()                                      # FastAPI 앱 인스턴스 생성
limiter = Limiter(key_func=get_remote_address)       # 클라이언트 IP 기준 속도 제한 설정
app.state.limiter = limiter                          # 앱에 속도 제한기 연결

# 속도 초과 시 예외 핸들러
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logging.warning(f"⛔ 속도 제한 초과: {request.client.host}")  # 로그 기록
    return JSONResponse(status_code=429, content={"error": "Too Many Requests"})  # 429 응답

# ------------------------- CORS 설정 -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                             # 모든 도메인에서 요청 허용
    allow_credentials=True,
    allow_methods=["*"],                             # 모든 HTTP 메서드 허용
    allow_headers=["*"],                             # 모든 헤더 허용
)

# ------------------------- 요청 모델 -------------------------
class ChatRequest(BaseModel):
    message: str                                     # 클라이언트로부터 받은 메시지 필드 정의

# ------------------------- /chat 엔드포인트 -------------------------
@app.post("/chat")
@limiter.limit("5/10seconds")                        # 10초에 5번 이상 요청 시 제한
async def chat(request: Request, body: ChatRequest):
    message = body.message.strip()                   # 메시지 앞뒤 공백 제거

    if not message:                                  # 빈 메시지일 경우 오류 응답
        return JSONResponse(status_code=400, content={"error": "메시지가 비어 있습니다."})

    logging.info(f"📩 요청 수신: {message}")          # 요청 로그 기록
    prompt = f"[INST] {message} [/INST]"             # 모델 입력 포맷 구성

    try:
        start_time = time.time()                     # 처리 시간 측정 시작
        response = llm(prompt, max_tokens=256, stop=["</s>"], echo=False)  # 모델 응답 요청
        elapsed = time.time() - start_time           # 처리 시간 계산

        # 모델 응답에서 선택지 추출
        choices = response.get("choices", [])
        if not choices or "text" not in choices[0]:  # 응답 형식이 이상할 경우 예외 발생
            raise ValueError("모델 응답 형식이 올바르지 않습니다.")

        answer = choices[0]["text"].strip()          # 응답 텍스트 추출 및 공백 제거
        logging.info(f"✅ 응답 완료 ({elapsed:.2f}s): {answer[:60]}...")  # 응답 로그

        return {"response": answer, "time_taken": round(elapsed, 2)}  # 응답 반환

    except Exception as e:
        logging.exception("❌ 에러 발생")            # 예외 발생 시 로그
        return JSONResponse(status_code=500, content={"error": str(e)})  # 500 에러 응답

# ------------------------- 서버 실행 -------------------------
# 로컬 서버 실행 함수 정의
def run():
    # FastAPI 앱 실행 (포트 8000, 외부 접속 허용)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    run()

