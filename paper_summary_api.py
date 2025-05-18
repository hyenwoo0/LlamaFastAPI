from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")

app = FastAPI()

# CORS 설정 (Unity, 웹에서도 접근 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 요약 함수
def summarize_text(text: str):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return "", ""

    # TF-IDF 기반 요약
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()
    top_tfidf_idx = tfidf_scores.argsort()[-2:][::-1]
    tfidf_summary = " ".join([sentences[i] for i in sorted(top_tfidf_idx)])

    # TextRank 기반 요약
    sim_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences) if len(s) > 30), reverse=True)
    text_rank_summary = " ".join([s for _, s in ranked_sentences[:2]])

    return tfidf_summary, text_rank_summary

# 논문 검색 및 요약 API
@app.get("/search")
def search_papers(keyword: str = Query(..., description="검색 키워드"), limit: int = 10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keyword,
        "offset": 0,
        "limit": limit,
        "fields": "title,abstract,authors,year,url"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {"error": f"논문 검색 실패 - status code: {response.status_code}"}

    data = response.json()
    papers = []

    for paper in data.get("data", []):
        abstract = paper.get("abstract", "")
        if not abstract or len(abstract) < 100:
            continue

        tfidf_sum, textrank_sum = summarize_text(abstract)
        authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])

        papers.append({
            "제목": paper.get("title", ""),
            "저자": authors,
            "발행년도": paper.get("year", ""),
            "초록": abstract,
            "링크": paper.get("url", ""),
            "TFIDF_요약": tfidf_sum,
            "TextRank_요약": textrank_sum
        })

    return papers

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_summary_api:app", host="0.0.0.0", port=8000, reload=True)

