from fastapi import FastAPI
from pydantic import BaseModel
from llm import get_ai_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="주택임대차법 챗봇 API")

# --- CORS 설정 ---
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://dive.oppspark.net",
    "https://dive.oppspark.net",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],   # 필요 시 ["GET","POST","OPTIONS"]로 제한 가능
    allow_headers=["*"],   # 필요 시 ["Content-Type","Authorization"] 등으로 제한 가능
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    response = ""
    for chunk in get_ai_response(query.question):
        response += chunk
    return {"answer": response}


@app.get("/")
def root():
    return {"status": "running5"}