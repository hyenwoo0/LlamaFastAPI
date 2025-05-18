# ------------------------- OS 및 로깅 관련 기본 모듈 임포트 -------------------------
import os
import logging
import time

# FastAPI 관련 모듈
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# SlowAPI 속도 제한
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Transformers 라이브러리: T5 모델
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ------------------------- 로그 설정 -------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "chat_server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    ]
)

# ------------------------- T5-base 모델 로딩 -------------------------
logging.info("🔄 T5-base 모델 로딩 중...")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logging.info("✅ T5-base 모델 로딩 완료")

# ------------------------- FastAPI 초기화 -------------------------
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logging.warning(f"⛔ 속도 제한 초과: {request.client.host}")
    return JSONResponse(status_code=429, content={"error": "Too Many Requests"})

# ------------------------- CORS 설정 -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- 요청 모델 -------------------------
class ChatRequest(BaseModel):
    message: str

# ------------------------- /chat 엔드포인트 -------------------------
@app.post("/chat")
@limiter.limit("5/10seconds")
async def chat(request: Request, body: ChatRequest):
    message = body.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "메시지가 비어 있습니다."})

    logging.info(f"📩 요청 수신: {message}")
    prompt = f"answer question: {message}"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=128)
        elapsed = time.time() - start_time

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"✅ 응답 완료 ({elapsed:.2f}s): {answer[:60]}...")

        return {
            "response": answer,
            "time_taken": round(elapsed, 2),
            "model": "t5-base"
        }

    except Exception as e:
        logging.exception("❌ 에러 발생")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------- 서버 실행 함수 -------------------------
def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ------------------------- 직접 실행 시 -------------------------
if __name__ == "__main__":
    run()
