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
import calendar

# ✅ NLTK punkt 다운로드 경로 설정
NLTK_DATA_PATH = os.path.expanduser("~/nltk_data")
nltk.data.path = [NLTK_DATA_PATH]  # 시스템 기본 경로 제거

# punkt 토크나이저가 없다면 다운로드
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DATA_PATH)

# ✅ FastAPI 인스턴스 생성
app = FastAPI()


# ✅ 초록 요약 함수: TextRank 방식 요약 (TF-IDF + PageRank)
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


# ✅ 키워드 추출 함수: TF-IDF 기준 상위 단어 추출
def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    if not text or len(text.strip()) < 10:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:top_n]]


# ✅ PubMed 논문 검색 및 정보 수집 함수
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

    # 월 문자열을 숫자형으로 변환하는 맵 (예: Oct → 10)
    month_map = {name: f"{num:02d}" for num, name in enumerate(calendar.month_abbr) if name}

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
                article_data = article.find(".//Article")

                # 제목
                title_en = article_data.findtext("ArticleTitle", default="(No Title)")

                # 초록
                abstract = article_data.findtext(".//AbstractText", default="")

                # 학술지 이름
                journal = article_data.findtext(".//Journal/Title", default="학술지 없음")

                # 발행일 구성
                pub_date_elem = article.find(".//PubDate")
                year = pub_date_elem.findtext("Year", "")
                month_raw = pub_date_elem.findtext("Month", "")
                day = pub_date_elem.findtext("Day", "01")
                month = month_map.get(month_raw[:3].capitalize(), "01") if month_raw else "01"
                pub_date = f"{year}-{month}-{day}" if year else "날짜 정보 없음"

                # 논문 유형
                article_type_list = article.findall(".//PublicationType")
                article_types = [a.text for a in article_type_list if a.text] or ["유형 없음"]

                # 페이지 정보
                pages = article.findtext(".//Pagination/MedlinePgn", default="페이지 정보 없음")

                # 저자 리스트 구성
                author_list = article_data.findall(".//Author")
                authors = []
                for author in author_list:
                    last = author.findtext("LastName")
                    fore = author.findtext("ForeName")
                    if last and fore:
                        authors.append(f"{fore} {last}")
                authors_str = authors if authors else ["저자 정보 없음"]

                # 요약 및 키워드 생성
                summary = summarize_abstract(abstract) if abstract else "No abstract available"
                keywords = extract_keywords(abstract) if abstract else []

                # 논문 링크 생성
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                # 논문 딕셔너리로 정리
                papers.append({
                    "title_en": title_en,
                    "abstract": summary,
                    "authors": authors_str,
                    "year": year or "연도 정보 없음",
                    "pub_date": pub_date,
                    "journal": journal,
                    "article_types": article_types,
                    "pages": pages,
                    "keywords": keywords,
                    "link": link
                })

                print(f"[✅] PMID {pmid} processed. Title: {title_en}")
                time.sleep(0.5)

        except Exception as e:
            print(f"[⚠️] Error fetching PMID {pmid}: {e}")
            continue

    return papers


# ✅ API 엔드포인트 정의 (/papers?query=...)
@app.get("/papers")
def get_paper_cards(query: str = Query(..., description="논문 검색 키워드")):
    try:
        papers = fetch_papers(query=query, retmax=5)
        return {"query": query, "papers": papers}
    except Exception as e:
        return {"error": str(e)}


# ✅ 실행 엔트리포인트 (로컬 실행 시 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("paper_summary_api:app", host="0.0.0.0", port=8000, reload=True)
