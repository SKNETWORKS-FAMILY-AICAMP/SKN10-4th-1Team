# Chapter 4: 벡터/그래프 검색 저장소 (Neo4jVectorSearch)


안녕하세요! 이전 [제3장: 언어 모델 연동 (GeminiLLM)](03_언어_모델_연동__geminillm__.md)에서는 우리 프로젝트의 똑똑한 AI 비서, `GeminiLLM`에 대해 배웠습니다. 이 AI 비서는 우리가 제공하는 정보를 바탕으로 사용자의 질문에 멋진 답변을 생성해 줄 수 있죠.

하지만 AI 비서가 아무리 똑똑해도, 정확하고 관련성 높은 '재료', 즉 연구 논문 정보를 제대로 찾아주지 못한다면 제대로 실력 발휘를 할 수 없을 거예요. 마치 도서관에 수많은 책이 있지만, 사서가 내가 찾는 주제와 *정말로* 관련된 책이나 특정 저자의 다른 중요한 책들을 콕 집어 찾아주지 못한다면 답답한 것과 같아요.

이번 장에서는 바로 이 '슈퍼 사서' 역할을 하는 **`Neo4jVectorSearch`**에 대해 알아볼 거예요. `Neo4jVectorSearch`는 사용자의 질문 의도에 맞춰, 단순히 키워드가 일치하는 논문을 넘어 **의미적으로 유사한 논문을 찾거나(벡터 검색)**, 논문들 간의 **숨겨진 연결 관계(그래프 검색)**를 탐색하여 AI 비서에게 최고의 정보를 전달하는 '전문 검색 엔진'입니다.

## Neo4jVectorSearch: 우리 프로젝트의 슈퍼 사서 📚🔍

`Neo4jVectorSearch`는 우리 연구 정보 시스템의 핵심 검색 기능을 담당합니다. 이 슈퍼 사서는 두 가지 특별한 능력을 가지고 있어요.

### 능력 1: 의미로 찾기 (벡터 검색 - Semantic Search) 🧠

혹시 "인공지능을 활용한 신약 개발 최신 동향"에 대해 알고 싶은데, 어떤 키워드로 검색해야 할지 막막했던 적 있으신가요? "AI 신약", "머신러닝 제약", "딥러닝 약물 발견"... 너무 많은 키워드가 떠오르죠.

**벡터 검색**은 이렇게 키워드만으로는 찾기 어려운, '의미'가 비슷한 정보를 찾아주는 기술입니다.
[제2장: 데이터 수집 및 전처리 (스크립트)](02_데이터_수집_및_전처리__스크립트__.md)에서 논문의 제목이나 초록을 **임베딩(embedding)**이라는 숫자 벡터로 변환했던 것을 기억하시나요? 임베딩은 텍스트의 의미를 숫자로 표현한 '좌표'와 같아요. 벡터 검색은 사용자의 질문도 임베딩으로 변환한 뒤, 이 질문 임베딩과 가장 가까운 '좌표'를 가진 논문 임베딩들을 찾아줍니다.

> **비유하자면?**
> 도서관 사서에게 "뭔가... 새롭고 희망찬 분위기의 소설을 찾고 있어요" 라고 말했을 때, 사서가 책 제목이나 저자 이름만 보는 게 아니라, 각 책의 전체적인 내용과 분위기를 파악해서 가장 비슷한 느낌의 책들을 추천해주는 것과 같아요!

`Neo4jVectorSearch`의 `semantic_search` 메서드가 바로 이 역할을 합니다. 사용자의 질문(쿼리)을 받아서 의미적으로 가장 유사한 논문들을 찾아주죠.

### 능력 2: 연결고리로 찾기 (그래프 검색 - Graph Search) 🕸️

어떤 논문을 읽다가 "이 논문을 쓴 다른 저자들은 또 어떤 연구를 했을까?" 혹은 "이 논문과 비슷한 주제를 다루는 다른 주요 논문들은 뭘까?" 궁금할 때가 있죠.

**그래프 검색**은 [제1장: 그래프 데이터 모델 (Neo4j 모델)](01_그래프_데이터_모델__neo4j_모델__.md)에서 배운 것처럼, Neo4j 데이터베이스에 저장된 논문, 저자, 키워드 간의 '관계'를 따라가며 정보를 찾는 기술입니다.

> **비유하자면?**
> 사서가 특정 책을 찾아주면서, "이 책의 저자가 쓴 다른 인기 시리즈도 있고요, 이 책에서 자주 언급되는 다른 중요한 책들도 이쪽에 있어요" 라며 관련된 책들을 줄줄이 꿰어 추천해주는 것과 같아요!

`Neo4jVectorSearch`에는 다음과 같은 그래프 검색 관련 메서드들이 있습니다:
*   `get_related_nodes`: 특정 논문(PMID)을 중심으로 그 논문의 저자, 키워드, 실린 학술지 정보를 찾아줍니다.
*   `find_article_connections`: 특정 논문이 다른 논문들과 어떤 연결고리(예: 공통 저자, 공통 키워드)를 통해 이어져 있는지 탐색합니다.

이 두 가지 능력을 통해 `Neo4jVectorSearch`는 단순 검색을 넘어, 정보들 사이의 맥락과 관계까지 파악하여 사용자에게 더욱 풍부하고 깊이 있는 검색 결과를 제공합니다.

## 코드 살펴보기: `Neo4jVectorSearch` 사용법 (`rag_pipeline/vector_store.py`)

그럼 이제 `Neo4jVectorSearch`를 어떻게 사용하는지 코드를 통해 간단히 살펴볼까요? 모든 마법은 `rag_pipeline/vector_store.py` 파일 안에 있는 `Neo4jVectorSearch` 클래스에 담겨 있습니다.

먼저 `Neo4jVectorSearch` 객체를 만들어야 합니다.

```python
# rag_pipeline/vector_store.py 파일에서 Neo4jVectorSearch를 가져옵니다.
from rag_pipeline.vector_store import Neo4jVectorSearch

# Neo4jVectorSearch 객체 생성
# "text-embedding-3-large"는 OpenAI의 임베딩 모델 이름입니다.
vector_store = Neo4jVectorSearch(embedding_model="text-embedding-3-large") 
```
*   객체를 만들 때 어떤 임베딩 모델을 사용할지 지정할 수 있습니다. 이 모델은 사용자의 질문을 임베딩으로 변환하는 데 사용됩니다.

### 예제 1: 의미로 논문 찾기 (`semantic_search`)

사용자가 "비만 치료에 대한 최신 연구"를 질문했다고 가정해봅시다. `semantic_search` 메서드를 사용하면 의미적으로 유사한 논문들을 찾을 수 있습니다.

```python
user_query = "비만 치료에 대한 최신 연구"
# top_k는 최대 몇 개의 결과를 받을지 지정합니다.
similar_articles = vector_store.semantic_search(query=user_query, top_k=5)

# 결과 출력 (예시)
for article in similar_articles:
    print(f"제목: {article['title']}")
    print(f"유사도: {article['similarity']:.4f}") # 소수점 4자리까지 표시
    print("-" * 20)
```
*   `semantic_search`는 질문(`query`)과 가져올 결과 수(`top_k`)를 인자로 받습니다.
*   결과로는 논문의 `pmid`, `title`, `abstract`, 그리고 질문과의 `similarity`(유사도 점수) 등이 담긴 딕셔너리 리스트를 반환합니다. 유사도 점수가 높을수록 질문과 의미적으로 가깝다는 뜻입니다.

**출력 예시 (어떤 결과가 나올까요?)**

```
제목: Recent Advances in Pharmacological Treatments for Obesity
유사도: 0.8765
--------------------
제목: Novel Therapeutic Targets for Obesity and Related Metabolic Disorders
유사도: 0.8543
--------------------
... (나머지 결과들) ...
```
위와 같이 사용자의 질문과 의미적으로 관련된 논문 목록과 각 논문의 유사도 점수를 얻을 수 있습니다.

### 예제 2: 특정 논문의 친구들 찾기 (`get_related_nodes`)

만약 `semantic_search`를 통해 찾은 특정 논문(예: PMID가 '12345678')에 대해 더 자세히 알고 싶다면, `get_related_nodes`를 사용할 수 있습니다.

```python
target_pmid = "12345678" # 예시 PMID
related_info = vector_store.get_related_nodes(pmid=target_pmid)

# 결과 출력 (예시)
print(f"--- {target_pmid} 논문의 관련 정보 ---")
print("저자:")
for author in related_info.get('authors', []):
    print(f"  - {author['full_name']}")
print("키워드:")
for keyword in related_info.get('keywords', []):
    print(f"  - {keyword['term']}")
if related_info.get('journal'):
    print(f"학술지: {related_info['journal']['name']}")
```
*   `get_related_nodes`는 논문의 고유 식별자인 `pmid`를 입력받습니다.
*   결과로는 해당 논문의 저자(authors), 키워드(keywords), 발행된 학술지(journal) 정보가 담긴 딕셔너리를 반환합니다.

### 예제 3: 논문 간의 다리 찾기 (`find_article_connections`)

특정 논문이 다른 논문들과 어떤 관계를 맺고 있는지 알고 싶을 때 `find_article_connections`를 사용합니다. 예를 들어, 공통 저자나 공통 키워드를 통해 연결된 다른 논문들을 찾아볼 수 있습니다.

```python
target_pmid = "12345678" # 예시 PMID
# degrees는 몇 단계까지 연결을 탐색할지 (현재 코드는 2 또는 3단계 고정)
connections = vector_store.find_article_connections(pmid=target_pmid) 

# 결과 출력 (예시)
print(f"--- {target_pmid} 논문과 연결된 다른 논문들 ---")
for conn in connections:
    print(f"연결된 논문 제목: {conn['title']}")
    print(f"  연결 방식: {conn['connection_description']}") # 예: "키워드 'Diabetes' 공유"
    print("-" * 10)
```
*   이 메서드는 특정 논문(`pmid`)과 연결된 다른 논문들의 정보(`pmid`, `title`, `abstract`)와 어떻게 연결되어 있는지(`connection_description`)를 알려줍니다.

## 더 똑똑한 검색: 검색 결과 개선하기 (리랭킹) 🥇

`semantic_search`로 찾은 결과는 이미 꽤 훌륭하지만, 때로는 더 나은 결과를 위해 한 단계 더 나아갈 수 있습니다. 바로 **리랭킹(Re-ranking)**입니다. 리랭킹은 초기 검색 결과를 가져온 뒤, 추가적인 정보나 다른 기준을 사용해서 순위를 다시 매기는 과정입니다.

우리 `Neo4jVectorSearch`는 두 가지 주요 리랭킹 전략을 사용할 수 있는 `two_stage_retrieval` (2단계 검색) 메서드를 제공합니다.

### 1단계: 번역 후 벡터 검색, 2단계: LLM으로 재평가 (`two_stage_retrieval`)

`two_stage_retrieval` 메서드는 이름처럼 두 단계로 검색을 수행하여 더 정확한 결과를 찾으려고 노력합니다.

1.  **1단계: (필요시) 쿼리 번역 및 벡터 검색**
    *   만약 사용자가 한국어로 질문했는데, 우리 논문 데이터의 제목/초록은 주로 영어이고 임베딩도 영어를 기반으로 만들어졌다면? `_translate_query_with_llm` 헬퍼 함수가 [제3장: 언어 모델 연동 (GeminiLLM)](03_언어_모델_연동__geminillm__.md)에서 배운 Gemini LLM을 사용해 한국어 질문을 영어로 번역합니다. 이렇게 하면 영어 기반 임베딩 공간에서 더 정확한 유사도 검색이 가능해집니다.
    *   번역된 (또는 원래 영어) 쿼리를 사용하여 `semantic_search`를 실행하여 초기 후보 논문 목록을 가져옵니다 (`initial_k` 만큼).

2.  **2단계: LLM 기반 리랭킹 (`_llm_only_reranking`)**
    *   1단계에서 찾은 초기 후보 논문들(예: 상위 10~20개)을 다시 한번 Gemini LLM에게 보여줍니다.
    *   LLM에게 원래의 **한국어 사용자 질문**과 각 후보 논문의 제목/초록을 주고, "이 논문이 사용자의 질문과 얼마나 관련이 깊은지 0점에서 10점 사이로 평가해주세요" 라고 요청합니다.
    *   LLM이 매긴 이 '관련성 점수'를 새로운 유사도 점수로 사용하여 최종 순위를 매깁니다 (`final_k` 만큼).

```python
user_query_korean = "청소년 당뇨병 예방을 위한 생활 습관 개선 연구"
# initial_k: 1단계에서 가져올 후보 수
# final_k: LLM 리랭킹 후 최종 결과 수
reranked_articles = vector_store.two_stage_retrieval(
    query=user_query_korean, 
    initial_k=20, # 초기 후보 20개
    final_k=5     # 최종 5개 선택
)

# 결과 출력
for article in reranked_articles:
    print(f"제목: {article['title']} (최종 유사도: {article['similarity']:.4f})")
```
이처럼 `two_stage_retrieval`은 먼저 넓게 검색한 후(벡터 검색), AI 비서(LLM)의 섬세한 판단을 통해 가장 적합한 결과를 골라내는 방식으로 검색 품질을 높입니다.

### (참고) 그래프 정보로 재정렬하기 (`graph_based_reranking`)

`Neo4jVectorSearch` 클래스에는 `graph_based_reranking`이라는 메서드도 있습니다. 이 방법은 벡터 검색으로 찾은 논문 목록을 가져온 뒤, 각 논문이 다른 논문들과 얼마나 많은 연결(예: 공통 저자, 공통 키워드)을 가지고 있는지, 또는 얼마나 많은 저자/키워드를 가지고 있는지 등의 **그래프 구조적 특징**을 분석하여 '그래프 점수'를 매깁니다. 그리고 이 그래프 점수와 원래의 벡터 유사도 점수를 결합하여 최종 순위를 매기는 방식입니다.
이 방법은 현재 `two_stage_retrieval`에서는 직접 사용되지 않지만, 그래프 정보를 활용한 리랭킹의 한 예시로 이해할 수 있습니다.

## Neo4jVectorSearch 내부 작동 원리 살짝 엿보기 ⚙️

그렇다면 `Neo4jVectorSearch`는 내부적으로 어떻게 이런 검색들을 수행할까요?

### 벡터 검색 (`semantic_search`)의 비밀

1.  **질문 임베딩**: 사용자의 질문 텍스트를 `create_embedding` 메서드를 통해 숫자 벡터(질문 임베딩)로 변환합니다.
    ```python
    # rag_pipeline/vector_store.py (Neo4jVectorSearch 클래스 내)
    # def create_embedding(self, text: str) -> List[float]:
    #     # ... (OpenAI API를 호출하여 임베딩 생성) ...
    #     response = client.embeddings.create(...)
    #     return response.data[0].embedding
    ```
2.  **논문 임베딩 가져오기**: Neo4j 데이터베이스에서 모든 `Article` 노드의 저장된 임베딩 벡터(`combined_embedding`)를 가져옵니다.
    ```cypher
    // Neo4j에서 실행되는 Cypher 쿼리 예시 (간략화)
    MATCH (a:Article)
    WHERE a.combined_embedding IS NOT NULL // 임베딩이 있는 논문만
    RETURN a.pmid, a.title, a.combined_embedding
    ```
3.  **유사도 계산**: 질문 임베딩과 데이터베이스에서 가져온 각 논문의 임베딩 벡터 간의 **코사인 유사도(cosine similarity)**를 계산합니다. 코사인 유사도는 두 벡터가 얼마나 비슷한 방향을 가리키는지 측정하는 방법으로, 0과 1 사이의 값을 가집니다 (1에 가까울수록 유사함).
    ```python
    # rag_pipeline/vector_store.py (Neo4jVectorSearch 클래스 내)
    # def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
    #     # ... (numpy를 사용하여 코사인 유사도 계산) ...
    #     # 참고: 저장된 임베딩이 문자열 형태일 수 있어 숫자 리스트로 변환하는 과정 포함
    #     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    ```
4.  **결과 정렬 및 반환**: 계산된 유사도 점수를 기준으로 논문들을 정렬하고, 가장 점수가 높은 상위 `top_k`개의 논문을 반환합니다.

이 과정을 순서도로 나타내면 다음과 같습니다:

```mermaid
sequenceDiagram
    participant UserQuery as 사용자 질문
    participant VectorStore as Neo4jVectorSearch
    participant OpenAI_Embed as OpenAI 임베딩 API
    participant Neo4jDB as Neo4j 데이터베이스
    participant SimilarityCalc as 유사도 계산 로직

    UserQuery->>VectorStore: semantic_search("질문 내용")
    VectorStore->>OpenAI_Embed: 질문 텍스트 임베딩 요청
    OpenAI_Embed-->>VectorStore: 질문 임베딩 벡터
    VectorStore->>Neo4jDB: 모든 논문 임베딩 요청 (Cypher)
    Neo4jDB-->>VectorStore: 논문 임베딩 목록
    VectorStore->>SimilarityCalc: (질문 임베딩, 각 논문 임베딩) 코사인 유사도 계산 요청
    SimilarityCalc-->>VectorStore: 유사도 점수들
    VectorStore-->>UserQuery: 상위 K개 유사 논문 목록 반환
end
```

### 그래프 검색 (`get_related_nodes`, `find_article_connections`)의 비밀

그래프 검색은 Neo4j의 강력한 쿼리 언어인 **Cypher**를 사용합니다. `get_related_nodes`나 `find_article_connections` 메서드 내부에서는 특정 `pmid`를 가진 `Article` 노드에서 시작하여, 정의된 관계(예: `AUTHORED_BY`, `HAS_KEYWORD`)를 따라 연결된 다른 노드들을 찾는 Cypher 쿼리를 실행합니다.

예를 들어, 특정 논문의 저자들을 찾는 `get_related_nodes`의 일부 Cypher 쿼리는 다음과 같이 생겼을 수 있습니다:

```cypher
// 특정 논문(pmid로 지정)의 저자 이름과 소속 가져오기
MATCH (author:Author)-[:AUTHORED_BY]->(a:Article {pmid: $pmid})
RETURN author.full_name as full_name, author.affiliation as affiliation
```
*   `$pmid`: 파이썬 코드에서 전달된 논문 PMID 값으로 대체됩니다.
*   `MATCH (author:Author)-[:AUTHORED_BY]->(a:Article {pmid: $pmid})`: "pmid가 $pmid인 Article 노드(a)를 작성한(AUTHORED_BY 관계) Author 노드(author)를 찾아라" 라는 의미입니다.

이처럼 Cypher 쿼리를 통해 그래프 데이터베이스의 풍부한 연결 정보를 효과적으로 탐색할 수 있습니다.

## 정리하며 🧐

이번 장에서는 우리 프로젝트의 '슈퍼 사서', `Neo4jVectorSearch`에 대해 알아보았습니다.

*   **벡터 검색**을 통해 사용자의 질문과 '의미적'으로 유사한 논문을 찾는 방법을 배웠습니다. (마치 사서가 책의 분위기까지 파악해 추천해주는 것처럼!)
*   **그래프 검색**을 통해 특정 논문과 연결된 저자, 키워드, 또는 다른 논문들을 찾는 방법을 배웠습니다. (마치 사서가 관련 도서를 줄줄이 꿰어 추천해주는 것처럼!)
*   `semantic_search`, `get_related_nodes`, `find_article_connections` 등 주요 메서드 사용법을 예제 코드로 살펴보았습니다.
*   `two_stage_retrieval`과 같이 LLM을 활용하여 검색 결과를 더욱 개선하는 리랭킹 개념도 맛보았습니다.
*   이러한 검색 기능들이 내부적으로 임베딩, 코사인 유사도, 그리고 Cypher 쿼리를 통해 어떻게 작동하는지 살짝 엿보았습니다.

이제 우리에게는 훌륭한 데이터 저장소([Neo4j](01_그래프_데이터_모델__neo4j_모델__.md), [데이터 수집](02_데이터_수집_및_전처리__스크립트__.md)), 그 내용을 이해하고 설명해줄 똑똑한 AI 비서([GeminiLLM](03_언어_모델_연동__geminillm__.md)), 그리고 이 비서에게 최적의 정보를 찾아줄 슈퍼 사서(`Neo4jVectorSearch`)까지 모두 준비되었습니다!

다음 장에서는 이 모든 구성 요소들을 어떻게 하나의 강력한 '질의응답 시스템', 즉 **RAG 파이프라인**으로 통합하여 사용자에게 최종적인 답변을 제공하는지 알아보겠습니다.

바로 [제5장: RAG 파이프라인 (HybridGraphFlow)](05_rag_파이프라인__hybridgraphflow__.md)에서 만나요!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)