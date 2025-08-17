from fastapi import FastAPI
from pydantic import BaseModel
from llm import get_ai_response

app = FastAPI()

class Query(BaseModel):
    question: str

# @app.post("/ask")
# def ask(query: Query):
#     response = ""
#     for chunk in get_ai_response(query.question):
#         response += chunk
#     return {"answer": response}

@app.post("/ask")
def ask(query: Query):
    # response = ""
    # for chunk in get_ai_response(query.question):
    #     response += chunk
    # return {"answer": response}
    return {"answer": f"질문 받음: {query.question}"}


@app.get("/")
def root():
    return {"status": "running"}