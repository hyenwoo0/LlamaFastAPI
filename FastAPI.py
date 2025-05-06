import os
import logging
import time
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from llama_cpp import Llama
import nest_asyncio
import uvicorn

# ------------------------- 로그 디렉토리 및 설정 -------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "chat_server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 로그
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 파일 로그
    ]
)

# ------------------------- 로컬 LLaMA 모델 로딩 -------------------------
MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"  # 모델 경로 설정
logging.info("🔄 모델 로딩 중...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
logging.info("✅ 모델 로딩 완료")

# ------------------------- FastAPI 앱 초기화 및 속도 제한 -------------------------
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logging.warning(f"⛔ 속도 제한 초과: {request.client.host}")
    return JSONResponse(status_code=429, content={"error": "Too Many Requests"})

# ------------------------- CORS 설정 (Unity 등 외부 접근 허용) -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- /chat API 엔드포인트 -------------------------
@app.post("/chat")
@limiter.limit("5/10seconds")  # IP당 10초에 최대 5회
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "").strip()

        if not message:
            return {"error": "메시지가 비어 있습니다."}

        logging.info(f"📩 요청 수신: {message}")
        prompt = f"[INST] {message} [/INST]"

        start_time = time.time()
        response = llm(prompt, max_tokens=256, stop=["</s>"], echo=False)
        elapsed = time.time() - start_time

        answer = response["choices"][0]["text"].strip()
        logging.info(f"✅ 응답 완료 ({elapsed:.2f}s): {answer[:60]}...")

        return {"response": answer, "time_taken": round(elapsed, 2)}

    except Exception as e:
        logging.exception("❌ 에러 발생")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------- 서버 실행 (Jupyter 대응 포함) -------------------------
nest_asyncio.apply()

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run, daemon=True).start()
