import os
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

# ✅ NLTK punkt 설정: 시스템 경로 제거 + 사용자 경로만 사용
NLTK_DATA_PATH = os.path.expanduser("~/nltk_data")
nltk.data.path = [NLTK_DATA_PATH]  # 🔥 경로 완전히 덮어쓰기 (기존 경로 제거)

try:
    nltk.data.find("tokenizers/punkt")
    _ = nltk.tokenize.punkt.PunktSentenceTokenizer().tokenize("Hello. World.")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_PATH)

app = FastAPI()

# ✅ 요약 함수
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

# ✅ 키워드 추출
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    if not text or len(text.strip()) < 10:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:top_n]]

# ✅ 논문 수집
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

    print("[🧪] PMID list:", id_list)

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    papers = []

    for pmid in id_list:
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        try:
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            root = ET.fromstring(fetch_response.text)

            articles = root.findall("PubmedArticle")
            for article in articles:
                title_en = article.findtext(".//ArticleTitle", default="(No Title)")
                abstract = article.findtext(".//AbstractText", default="")

                pub_date = article.find(".//PubDate")
                year = pub_date.findtext("Year") if pub_date is not None else None
                if not year:
                    year = pub_date.findtext("MedlineDate", default="Unknown") if pub_date is not None else "Unknown"

                authors = [
                    f"{a.findtext('ForeName', '')} {a.findtext('LastName', '')}".strip()
                    for a in article.findall(".//Author")
                    if a.find("LastName") is not None
                ]

                summary = summarize_abstract(abstract) if abstract else "No abstract available"
                keywords = extract_keywords(abstract) if abstract else []

                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                papers.append({
                    "title_en": title_en,
                    "abstract": summary,
                    "authors": authors,
                    "year": year,
                    "keywords": keywords,
                    "link": link
                })

                print(f"[✅] PMID {pmid} processed. Title: {title_en}")
                time.sleep(0.5)

        except Exception as e:
            print(f"[⚠️] Error fetching PMID {pmid}: {e}")
            continue

    return papers

# ✅ API 엔드포인트
@app.get("/papers")
def get_paper_cards(query: str = Query(..., description="논문 검색 키워드")):
    try:
        papers = fetch_papers(query=query, retmax=5)
        return {"query": query, "papers": papers}
    except Exception as e:
        return {"error": str(e)}

# ✅ 실행 진입점
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_summary_api:app", host="0.0.0.0", port=8000, reload=True)
