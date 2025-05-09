# Chapter 3: Neo4j 데이터 모델 및 연동


안녕하세요! `SKN10-4th-1Team` 프로젝트 튜토리얼의 세 번째 장에 오신 것을 환영합니다. 지난 [Chapter 2: 하이브리드 RAG 파이프라인](02_하이브리드_rag_파이프라인_.md)에서는 사용자의 질문 의도에 맞춰 벡터 검색, 그래프 검색, 또는 두 가지를 혼합한 하이브리드 검색을 수행하여 LLM에게 최적의 정보를 전달하는 과정을 배웠습니다. 이때 "그래프 검색"이 등장했었죠? 이번 장에서는 바로 그 그래프 검색의 핵심이 되는 **Neo4j 데이터베이스의 모델**은 어떻게 생겼고, 우리 시스템이 이 데이터베이스와 **어떻게 정보를 주고받는지(연동)** 자세히 알아볼 거예요. 복잡하게 얽힌 의학 연구 정보들을 어떻게 효과적으로 표현하고 탐색하는지 함께 살펴봅시다!

## 데이터, 점과 선으로 연결하기: Neo4j는 왜 필요할까요?

혹시 이런 궁금증을 가져본 적 있으신가요? "특정 논문 A의 저자들이 과거에 어떤 다른 연구들을 주로 진행했는지 알고 싶어." 또는 "논문 A가 다루는 핵심 주제(키워드)들이 다른 어떤 논문들과 연관되어 있는지 한눈에 보고 싶어!" 이런 질문들은 단순한 키워드 검색만으로는 답을 찾기 어렵습니다. 정보들 사이의 **'관계'**를 파악하는 것이 중요하기 때문이죠.

바로 이럴 때 **그래프 데이터베이스**인 Neo4j가 빛을 발합니다! Neo4j는 마치 복잡한 관계도를 그리는 것처럼 데이터를 저장하고 탐색할 수 있게 해줍니다.
우리 시스템에서는 논문, 저자, 키워드, 저널 같은 의학 연구 정보들을 **점(노드)**으로, 그리고 이들 사이의 관계(예: "저자가 논문을 작성했다", "논문이 특정 키워드를 포함한다")를 **선(관계)**으로 표현합니다.

예를 들어,
*   **점**: "비만과 당뇨병"이라는 논문, "김 연구"라는 저자, "인슐린 저항성"이라는 키워드
*   **선**: "김 연구" --- (작성함) ---> "비만과 당뇨병" --- (다루는 주제) ---> "인슐린 저항성"

이렇게 점과 선으로 연결된 데이터 구조 덕분에, 우리는 특정 정보를 중심으로 그와 연결된 다른 정보들을 마치 거미줄처럼 따라가며 탐색할 수 있습니다. "김 연구 저자가 참여한 다른 모든 논문은 무엇일까?", "인슐린 저항성 키워드를 가진 다른 논문들은 어떤 저자들이 썼을까?" 같은 질문에 대한 답을 효과적으로 찾을 수 있죠.

이 장에서는 Neo4j에 우리 연구 정보들이 어떤 모습(데이터 모델)으로 저장되는지, 그리고 우리 파이썬 시스템이 Neo4j와 어떻게 대화하는지(연동)를 쉽고 재미있게 알아보겠습니다.

## Neo4j 데이터 모델의 핵심 요소들: 정보의 설계도

Neo4j에 데이터를 저장하기 전에, 우리는 먼저 정보들을 어떤 모양으로 구성할지 설계도를 그려야 합니다. 이 설계도를 **데이터 모델**이라고 부릅니다. 우리 시스템의 데이터 모델은 크게 다음과 같은 요소들로 이루어져 있어요.

### 1. 점 (노드, Node): 우리 정보의 주인공들

노드는 그래프에서 가장 기본적인 정보 단위를 나타냅니다. 우리 시스템에서는 다음과 같은 정보들이 각각의 노드가 됩니다.

*   **논문 (Article)**: 개별 연구 논문 하나하나를 의미합니다. (예: "코로나19 백신 효과 분석")
*   **저자 (Author)**: 논문을 작성한 연구자입니다. (예: "홍길동 박사")
*   **키워드 (Keyword)**: 논문의 핵심 주제어나 내용을 나타내는 단어입니다. (예: "mRNA 백신", "면역 반응")
*   **저널 (Journal)**: 논문이 실린 학술지입니다. (예: "Nature Medicine")

파이썬 코드에서는 `neomodel`이라는 라이브러리를 사용하여 이런 노드들을 클래스로 정의합니다. 예를 들어 `Article` 노드는 `api/models.py` 파일에 다음과 같이 정의될 수 있습니다 (간략화된 버전):

```python
# api/models.py (Article 모델 일부)
from neomodel import StructuredNode, StringProperty, JSONProperty # 필요한 도구들

class Article(StructuredNode): # 'Article'이라는 종류의 점(노드)을 정의
    # Article 노드가 가질 수 있는 정보들 (속성)
    pmid = StringProperty(unique_index=True, required=True) # 논문 고유번호 (필수!)
    title = StringProperty(required=True) # 논문 제목 (필수!)
    abstract = StringProperty()           # 논문 초록
    # ... 다른 정보들 ...
```
위 코드에서 `Article` 클래스는 "나는 Article이라는 종류의 노드(점)야!" 라고 선언하는 것과 같습니다. `pmid`, `title` 등은 이 Article 노드가 가질 수 있는 세부 정보(속성)들입니다.

### 2. 선 (관계, Relationship): 주인공들을 잇는 이야기

관계는 노드들 사이의 연결을 의미하며, 이 연결이 어떤 의미를 가지는지 나타냅니다. "누가 무엇을 했다", "무엇이 무엇에 속한다" 와 같은 정보들이죠.

*   **AUTHORED_BY (저술)**: 저자 노드와 논문 노드를 연결하며, '저자가 이 논문을 작성했다'는 의미입니다. (예: 홍길동 ---AUTHORED_BY---> "코로나19 백신 효과 분석")
*   **HAS_KEYWORD (포함)**: 논문 노드와 키워드 노드를 연결하며, '논문이 이 키워드를 포함한다'는 의미입니다. (예: "코로나19 백신 효과 분석" ---HAS_KEYWORD---> "mRNA 백신")
*   **PUBLISHED_IN (게재)**: 논문 노드와 저널 노드를 연결하며, '논문이 이 저널에 게재되었다'는 의미입니다. (예: "코로나19 백신 효과 분석" ---PUBLISHED_IN---> "Nature Medicine")

`api/models.py` 파일에서는 이러한 관계들을 다음과 같이 정의합니다. 예를 들어 `Author` 노드가 `Article` 노드와 `AUTHORED_BY` 관계로 연결되는 부분입니다.

```python
# api/models.py (Author 모델 일부)
from neomodel import StructuredNode, StringProperty, RelationshipTo # 필요한 도구들

class Author(StructuredNode): # 'Author'라는 종류의 점(노드)을 정의
    full_name = StringProperty(index=True, required=True) # 저자 전체 이름
    # ... 다른 정보들 ...

    # 'Article' 노드로 향하는 'AUTHORED_BY' 라는 이름의 관계를 정의
    articles = RelationshipTo('Article', 'AUTHORED_BY') 
```
여기서 `RelationshipTo('Article', 'AUTHORED_BY')`는 "Author 노드는 Article 노드로 'AUTHORED_BY' 라는 이름의 선(관계)을 가질 수 있어!" 라고 말하는 것과 같습니다.

### 3. 딱지 (레이블, Label): 주인공의 종류

레이블은 노드의 '종류'나 '분류'를 나타냅니다. 예를 들어, 어떤 노드가 '논문'인지, '저자'인지, '키워드'인지를 구분하는 딱지 같은 것이죠. 앞서 `Article`, `Author` 클래스를 정의한 것 자체가 Neo4j에게 "이 클래스로 만들어진 노드들은 각각 'Article', 'Author'라는 레이블을 붙여줘!" 라고 알려주는 것입니다. Neo4j는 이 레이블을 보고 특정 종류의 노드들만 빠르게 찾아낼 수 있습니다.

### 4. 이름표 (속성, Property): 주인공과 선의 상세 정보

속성은 노드나 관계가 가질 수 있는 구체적인 데이터 값들입니다. 마치 사람에게 이름, 나이, 주소 같은 정보가 있듯이, 노드와 관계도 자신만의 상세 정보를 가질 수 있습니다.

*   **노드 속성**:
    *   `Article` 노드: `pmid` (논문 고유번호), `title` (제목), `publication_year` (출판 연도) 등
    *   `Author` 노드: `full_name` (전체 이름), `affiliation` (소속 기관) 등
    *   `Keyword` 노드: `term` (키워드 용어) 등
*   **관계 속성 (선택 사항)**: 관계 자체도 속성을 가질 수 있습니다. 예를 들어, 저자와 논문 사이의 `AUTHORED_BY` 관계에 `order` (저자 순서) 같은 속성을 추가할 수도 있습니다. (우리 프로젝트에서는 현재 관계 속성을 많이 사용하지는 않습니다.)

`api/models.py`의 클래스 정의에서 `StringProperty`, `IntegerProperty` 등으로 선언된 것들이 바로 이 속성들입니다.

### 5. 특별한 정보: 임베딩 벡터 저장

우리 시스템은 논문의 제목이나 초록을 숫자로 이루어진 긴 배열, 즉 **임베딩 벡터**로 변환하여 저장합니다. 이 임베딩 벡터는 [Chapter 2: 하이브리드 RAG 파이프라인](02_하이브리드_rag_파이프라인_.md)에서 배운 '의미 기반 검색(벡터 검색)'에 사용되어, 사용자의 질문과 의미적으로 유사한 논문을 찾는 데 핵심적인 역할을 합니다.

`Article` 노드는 이 임베딩 벡터를 저장하기 위한 특별한 속성을 가집니다. `api/models.py`에서는 `JSONProperty`를 사용하여 리스트 형태의 숫자 배열(벡터)을 저장합니다.

```python
# api/models.py (Article 모델 일부)
from neomodel import JSONProperty # JSON 형태로 데이터를 저장하는 속성

class Article(StructuredNode):
    # ... 다른 속성들 ...
    title_embedding = JSONProperty()      # 제목 임베딩 벡터
    abstract_embedding = JSONProperty()   # 초록 임베딩 벡터
    combined_embedding = JSONProperty()   # 제목+초록 통합 임베딩 벡터
```
이렇게 저장된 임베딩 벡터 덕분에, Neo4j는 의미적으로 관련된 논문들을 빠르게 찾아낼 수 있는 강력한 검색 능력을 갖추게 됩니다.

### 우리 시스템의 데이터 모델 한눈에 보기

이 모든 요소들을 종합하면, 우리 시스템의 Neo4j 데이터 모델은 다음과 같이 그림으로 표현할 수 있습니다.

```mermaid
graph LR
    subgraph SKN10_4th_1Team 의학 논문 데이터 모델
        A(Author<br>저자) -- AUTHORED_BY<br>(저술 관계) --> Ar(Article<br>논문);
        Ar -- PUBLISHED_IN<br>(게재 관계) --> J(Journal<br>저널);
        Ar -- HAS_KEYWORD<br>(키워드 관계) --> K(Keyword<br>키워드);
    end
```
*   **저자(Author)** 는 논문(Article)을 **저술(AUTHORED_BY)** 합니다.
*   **논문(Article)** 은 특정 저널(Journal)에 **게재(PUBLISHED_IN)** 됩니다.
*   **논문(Article)** 은 여러 키워드(Keyword)를 **포함(HAS_KEYWORD)** 합니다.

이런 구조 덕분에, 예를 들어 특정 저자로부터 시작해서 그 저자가 쓴 논문들을 찾고, 그 논문들이 어떤 키워드를 가지는지, 어떤 저널에 실렸는지 등을 연쇄적으로 탐색할 수 있게 됩니다.

## Neo4j와 우리 시스템의 만남: 어떻게 연결될까요?

자, 이제 데이터 모델이라는 설계도는 준비되었습니다. 그렇다면 우리 파이썬으로 만들어진 시스템은 이 설계도를 바탕으로 어떻게 Neo4j 데이터베이스와 실제로 정보를 주고받을까요?

### 1. 데이터 모델 설계 코드로 옮기기: `api/models.py`

앞서 살펴본 것처럼, 우리는 `neomodel`이라는 아주 유용한 파이썬 라이브러리를 사용합니다. 이 라이브러리 덕분에 Neo4j의 노드와 관계를 파이썬 클래스로 쉽게 정의할 수 있습니다. `api/models.py` 파일이 바로 이 역할을 하는 곳입니다.

예를 들어, `Article` 노드와 그 속성, 그리고 다른 노드와의 관계는 다음과 같이 파이썬 코드로 표현됩니다.

```python
# api/models.py (Article 모델 일부)
from neomodel import (StructuredNode, StringProperty, IntegerProperty, 
                        JSONProperty, RelationshipTo, RelationshipFrom)

class Article(StructuredNode):
    # 속성들 (노드의 정보)
    pmid = StringProperty(unique_index=True, required=True) # 논문 고유 ID (필수, 중복 불가)
    title = StringProperty(required=True)                  # 논문 제목 (필수)
    publication_year = IntegerProperty(index=True)         # 출판 연도
    combined_embedding = JSONProperty()                    # 통합 임베딩 벡터

    # 관계들 (다른 노드와의 연결선)
    # 'Author' 노드로부터 'AUTHORED_BY' 관계를 통해 연결됨
    authors = RelationshipFrom('Author', 'AUTHORED_BY') 
    # 'Journal' 노드로 'PUBLISHED_IN' 관계를 통해 연결됨
    journal = RelationshipTo('Journal', 'PUBLISHED_IN')
    # 'Keyword' 노드로 'HAS_KEYWORD' 관계를 통해 연결됨
    keywords = RelationshipTo('Keyword', 'HAS_KEYWORD')

    def __str__(self): # 객체를 문자열로 표현할 때의 형식
        return f"{self.title} (PMID: {self.pmid})"
```
이렇게 파이썬 클래스로 모델을 정의하면, `neomodel`이 내부적으로 Neo4j가 이해할 수 있는 방식으로 변환해줍니다. 마치 번역가처럼요!

### 2. Neo4j 데이터베이스 연결 설정 (`api/models.py` 또는 `config/settings.py`)

우리 파이썬 시스템이 Neo4j 데이터베이스와 "대화"하려면, 먼저 연결 통로부터 만들어야겠죠? 이 연결 정보는 주로 `.env` 파일에 안전하게 보관하고, 시스템이 시작될 때 이 정보를 읽어와 Neo4j에 접속합니다.

`api/models.py` 파일 상단이나 `config/settings.py` 파일에서 `neomodel`에게 Neo4j 서버가 어디에 있는지 알려주는 코드를 볼 수 있습니다.

```python
# api/models.py 상단 또는 config/settings.py
from neomodel import config # neomodel 설정을 위한 도구
import os                  # 환경 변수를 읽기 위한 도구
from dotenv import load_dotenv # .env 파일 로드용

load_dotenv() # .env 파일에서 환경 변수 로드

# .env 파일에 정의된 NEO4J_BOLT_URL 값을 가져와서 연결 정보로 사용
# 예: 'bolt://neo4j사용자이름:비밀번호@데이터베이스주소:포트번호'
NEO4J_BOLT_URL = os.getenv('NEO4J_BOLT_URL', 'bolt://neo4j:your_password@localhost:7687')
config.DATABASE_URL = NEO4J_BOLT_URL # neomodel에 데이터베이스 주소 알려주기

print(f"Neomodel 연결 설정: {config.DATABASE_URL}")
```
이 설정 덕분에 우리 시스템은 `neomodel`을 통해 Neo4j 데이터베이스에 명령을 내리거나 데이터를 가져올 수 있게 됩니다.

### 3. 데이터 창고에 정보 쌓기 (데이터 적재 - `scripts/ingest_data.py`)

모델도 만들고 연결도 했으니, 이제 실제 데이터를 Neo4j에 차곡차곡 쌓아야 합니다. 이 과정은 주로 `scripts/ingest_data.py` 같은 스크립트를 통해 이루어집니다. 이 스크립트는 외부 데이터 소스(예: PubMed XML 파일)에서 논문 정보를 읽어와, 앞서 정의한 `Article`, `Author` 등의 `neomodel` 클래스를 사용하여 Neo4j에 노드와 관계를 생성하고 저장합니다.

데이터 적재 과정은 [Chapter 4: 데이터 수집 및 전처리 스크립트](04_데이터_수집_및_전처리_스크립트_.md)에서 더 자세히 다룰 예정이지만, 아주 간단한 예시를 통해 `neomodel`로 어떻게 데이터를 저장하는지 맛보기로 보여드릴게요.

```python
# scripts/ingest_data.py (매우 간략화된 가상 예시)
from api.models import Article, Author # 우리가 정의한 모델(클래스) 가져오기

# 1. '인공지능과 의학'이라는 제목의 새 논문 노드 만들기
new_article = Article(pmid="12345", title="인공지능과 의학")
new_article.save() # Neo4j에 저장!

# 2. '김연구'라는 이름의 새 저자 노드 만들기
new_author = Author(full_name="김연구")
new_author.save() # Neo4j에 저장!

# 3. '김연구' 저자가 '인공지능과 의학' 논문을 썼다는 관계 연결하기
# Author 모델의 'articles' 관계를 사용 (Author -AUTHORED_BY-> Article)
new_author.articles.connect(new_article)
```
`save()` 메소드는 노드를 데이터베이스에 저장하고, `connect()` 메소드는 두 노드 사이에 정의된 관계를 만들어줍니다. 정말 간단하죠?

### 4. 정보 탐험 언어, 사이퍼 (Cypher) 쿼리

데이터가 Neo4j에 잘 저장되었다면, 이제 필요한 정보를 꺼내 써야 합니다. Neo4j에서 데이터를 검색하고 조작할 때 사용하는 특별한 언어가 있는데, 바로 **사이퍼(Cypher)**입니다. 사이퍼는 마치 관계형 데이터베이스에서 SQL을 사용하듯, 그래프 데이터베이스를 위한 질문 언어라고 생각할 수 있습니다.

사이퍼는 점(노드)과 선(관계)을 ASCII 아트로 표현하는 것처럼 직관적인 문법을 가지고 있어서 배우기 쉽습니다. 예를 들어, "홍길동이라는 저자가 작성한 모든 논문의 제목을 찾아줘"라는 질문은 다음과 같은 사이퍼 쿼리로 표현될 수 있습니다.

```cypher
-- "홍길동" 이라는 이름을 가진 Author 노드를 찾고,
-- 그 Author가 AUTHORED_BY 관계로 연결된 Article 노드를 찾아서,
-- 그 Article 노드의 title 속성을 반환해줘!
MATCH (author:Author {full_name: "홍길동"})-[:AUTHORED_BY]->(article:Article)
RETURN article.title
```
`MATCH`는 "이런 패턴을 찾아줘"라는 의미이고, `( )`는 노드를, `-[ ]->`는 관계를 나타냅니다.

우리 시스템에서는 [Chapter 1: LLM 통합 (응답 생성 및 쿼리 이해)](01_llm_통합__응답_생성_및_쿼리_이해__.md)에서 배운 것처럼 LLM이 사용자의 자연어 질문을 이해하고, 이러한 사이퍼 쿼리를 동적으로 생성하기도 합니다! 또는, `rag_pipeline/vector_store.py` 와 같은 파일에서 미리 정의된 사이퍼 쿼리를 사용하여 특정 정보를 가져오기도 합니다.

## 시스템 내부에서는 무슨 일이? 정보 탐색 과정 엿보기

그렇다면 사용자가 질문을 했을 때, 시스템 내부에서는 Neo4j와 어떻게 상호작용하며 정보를 찾아낼까요?

### 1. 요청부터 응답까지: 간단한 흐름도

사용자의 질문이 그래프 데이터베이스 검색을 필요로 할 때, 대략 다음과 같은 흐름으로 진행됩니다.

```mermaid
sequenceDiagram
    participant 사용자
    participant 시스템_내부 as 시스템 (내부 로직)
    participant LLM_도우미 as LLM (사이퍼 생성)
    participant Neo4j_창고 as Neo4j 데이터베이스
    
    사용자->>시스템_내부: "코로나 백신 관련 연구한 '김박사'의 논문 찾아줘"
    activate 시스템_내부
    시스템_내부->>LLM_도우미: 질문 분석 및 사이퍼 생성 요청
    activate LLM_도우미
    LLM_도우미-->>시스템_내부: 생성된 사이퍼 쿼리 전달 (예: MATCH (a:Author {name:'김박사'})-[:AUTHORED_BY]->(p:Article)-[:HAS_KEYWORD]->(:Keyword {term:'코로나 백신'}) RETURN p.title)
    deactivate LLM_도우미
    시스템_내부->>Neo4j_창고: 사이퍼 쿼리 실행 요청
    activate Neo4j_창고
    Neo4j_창고-->>시스템_내부: 검색 결과 (논문 제목 리스트 등)
    deactivate Neo4j_창고
    시스템_내부->>사용자: 결과 바탕으로 답변 생성 및 전달
    deactivate 시스템_내부
```
1.  사용자가 질문을 합니다.
2.  시스템은 질문을 분석하고, 만약 그래프 정보가 필요하다고 판단되면 LLM에게 사이퍼 쿼리 생성을 요청할 수 있습니다. (또는 미리 준비된 쿼리 사용)
3.  생성된 (또는 준비된) 사이퍼 쿼리를 Neo4j 데이터베이스에 전달하여 실행합니다.
4.  Neo4j는 쿼리 결과(예: 논문 목록, 저자 정보 등)를 반환합니다.
5.  시스템은 이 결과를 바탕으로 사용자에게 최종 답변을 만들어 보여줍니다.

### 2. 코드 속 그래프 탐험: `rag_pipeline/vector_store.py`

실제 코드에서는 `neomodel` 라이브러리의 `db.cypher_query()` 함수를 사용하여 Neo4j에 직접 사이퍼 쿼리를 실행하고 결과를 받아올 수 있습니다. `rag_pipeline/vector_store.py` 파일에는 이러한 방식으로 Neo4j와 통신하여 특정 논문과 관련된 저자, 키워드 등의 정보를 가져오는 함수들이 구현되어 있습니다.

예를 들어, 특정 PMID를 가진 논문의 저자 이름을 찾는 함수는 다음과 같이 만들 수 있습니다.

```python
# rag_pipeline/vector_store.py (함수 예시)
from neomodel import db # Neo4j 데이터베이스 직접 접근 도구

def find_authors_of_article(article_pmid: str):
    # 사이퍼 쿼리: 특정 PMID를 가진 논문의 저자 이름 찾기
    # $pmid_param 부분은 아래 cypher_query 함수 호출 시 채워질 값입니다.
    cypher_query_template = """
    MATCH (author:Author)-[:AUTHORED_BY]->(a:Article {pmid: $pmid_param})
    RETURN author.full_name
    """
    # 쿼리 실행 및 결과 받기
    # {'pmid_param': article_pmid} 부분이 $pmid_param에 실제 값을 전달합니다.
    results, _ = db.cypher_query(cypher_query_template, {'pmid_param': article_pmid})
    
    # 결과에서 저자 이름만 추출하여 리스트로 반환
    # results는 [(이름1,), (이름2,), ...] 형태의 리스트일 수 있습니다.
    author_names = [row[0] for row in results] # 각 row의 첫 번째 값을 가져옴
    return author_names

# 사용 예시: pmid가 "12345"인 논문의 저자 찾기
# authors = find_authors_of_article("12345")
# print(authors) # 결과: ['김연구', '이박사'] (실제 데이터에 따라 다름)
```
이처럼 `db.cypher_query()` 함수는 우리 시스템이 Neo4j 데이터베이스와 직접 "대화"할 수 있는 중요한 통로 역할을 합니다. 이 함수를 통해 복잡한 관계를 탐색하거나, 특정 조건을 만족하는 노드 및 관계 정보를 효과적으로 가져올 수 있습니다.

## 정리하며

이번 장에서는 우리 `SKN10-4th-1Team` 프로젝트의 심장부 중 하나인 **Neo4j 데이터 모델**과 우리 **시스템과의 연동 방식**에 대해 알아보았습니다. 논문, 저자, 키워드와 같은 의학 연구 정보들이 어떻게 점(노드)과 선(관계)으로 표현되어 Neo4j에 저장되는지, 그리고 `neomodel` 라이브러리와 사이퍼 쿼리를 통해 이 정보들을 어떻게 활용하는지 살펴보았습니다. 임베딩 벡터를 노드에 함께 저장하여 의미 검색과 관계 검색을 모두 지원하는 기반을 마련한 것도 확인했죠!

이렇게 잘 설계된 데이터 모델과 효율적인 연동 방식은 우리 시스템이 사용자의 복잡한 질문에도 깊이 있는 답변을 제공할 수 있게 하는 중요한 밑거름이 됩니다.

하지만, 이렇게 멋진 데이터 모델도 결국은 실제 데이터가 잘 채워져 있어야 그 가치를 발휘할 수 있습니다. 그렇다면 이 많은 논문 데이터들은 어디서 가져와서, 어떤 과정을 거쳐 Neo4j에 저장되는 걸까요?

다음 장인 [Chapter 4: 데이터 수집 및 전처리 스크립트](04_데이터_수집_및_전처리_스크립트_.md)에서는 바로 이 데이터들을 수집하고, 우리 모델에 맞게 가공하여 Neo4j에 차곡차곡 쌓아나가는 과정, 즉 데이터 파이프라인의 첫 단계를 자세히 탐험해 볼 예정입니다. 어떤 마법 같은 스크립트들이 이 모든 일을 처리하는지 함께 알아보아요!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)