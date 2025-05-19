import requests
import time
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Query
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np

nltk.download("punkt")

app = FastAPI()

# ✅ 요약 함수 (TextRank)
def summarize_abstract(abstract: str, num_sentences: int = 2) -> str:
    sentences = sent_tokenize(abstract)
    if len(sentences) < 2:
        return abstract

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])
    return summary

# ✅ 키워드 추출 함수 (TF-IDF 기반)
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    if not text or len(text.strip()) < 10:
        return []

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in ranked[:top_n]]
    return keywords

# ✅ PubMed 논문 검색 및 처리
def fetch_papers(query: str, retmax: int = 5) -> List[dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json"
    }
    response = requests.get(search_url, params=search_params)
    id_list = response.json().get("esearchresult", {}).get("idlist", [])

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    papers = []

    for pmid in id_list:
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        root = ET.fromstring(fetch_response.text)

        article = root.find(".//PubmedArticle")
        if article is None:
            continue

        title_en = article.findtext(".//ArticleTitle", default="")
        abstract = article.findtext(".//AbstractText", default="")
        summary = summarize_abstract(abstract) if abstract else ""
        keywords = extract_keywords(abstract) if abstract else []
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        papers.append({
            "title_kr": "예시 번역된 제목입니다",  # 실제 번역은 추후 적용 가능
            "title_en": title_en,
            "abstract": summary,
            "keywords": keywords,
            "link": link
        })

        if len(papers) >= retmax:
            break

        time.sleep(0.5)

    return papers

# ✅ API 엔드포인트
@app.get("/papers")
def get_paper_cards(query: str = Query(..., description="논문 검색 키워드")):
    try:
        papers = fetch_papers(query=query, retmax=5)
        return {"query": query, "papers": papers}
    except Exception as e:
        return {"error": str(e)}

# 기존 코드 끝에 추가
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_summary_api:app", host="0.0.0.0", port=8000, reload=True)

