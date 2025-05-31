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

# Transformers 라이브러리: MT5 모델
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
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

# ------------------------- MT5 모델 로딩 -------------------------
HF_MODEL_ID = "pleyel/chatbot_test"

logging.info("🔄 Hugging Face 모델 로딩 중...")
tokenizer = MT5Tokenizer.from_pretrained(HF_MODEL_ID)
model = MT5ForConditionalGeneration.from_pretrained(HF_MODEL_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
logging.info("✅ 모델 로딩 완료")

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

@app.post("/chat")
@limiter.limit("5/10seconds")
async def chat(request: Request, body: ChatRequest):
    message = body.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "메시지가 비어 있습니다."})

    logging.info(f"📩 요청 수신: {message}")

    try:
        inputs = tokenizer(
            message,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        ).to(device)

        max_output_len = 64  # ✅ 출력 길이 하드코딩 또는 상단에서 따로 설정해도 OK

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_len,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3
            )
        elapsed = time.time() - start_time

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 반복 문장 필터링
        tokens = answer.split()
        if len(tokens) > 6 and tokens[:3] == tokens[3:6]:
            answer = "답변이 반복되어 정확히 인식되지 않았습니다. 다시 질문해 주세요."

        logging.info(f"✅ 응답 완료 ({elapsed:.2f}s): {answer[:60]}...")

        return {
            "response": answer,
            "time_taken": round(elapsed, 2),
            "model": HF_MODEL_ID
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
