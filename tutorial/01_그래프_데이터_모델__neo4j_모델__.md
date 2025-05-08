# Chapter 1: 그래프 데이터 모델 (Neo4j 모델)


안녕하세요! `SKN10-4th-1Team` 프로젝트의 첫 번째 여정에 오신 것을 환영합니다. 앞으로 몇 개의 튜토리얼을 통해 저희 프로젝트가 어떻게 연구 정보를 효과적으로 관리하고 탐색하는지 함께 알아볼 거예요.

연구의 세계는 정보들로 가득 차 있습니다. 수많은 논문, 그 논문을 쓴 저자들, 논문의 핵심 내용을 담은 키워드, 그리고 논문이 실린 학술지까지... 이 모든 정보들은 서로 복잡하게 얽혀 있죠. 예를 들어, "특정 저자가 어떤 논문들을 썼고, 그 논문들은 주로 어떤 키워드들을 다루고 있을까?" 혹은 "A라는 키워드를 가진 논문들은 주로 어떤 학술지에 실렸을까?" 같은 질문에 답을 찾고 싶을 때가 많습니다.

이런 복잡한 관계를 효과적으로 표현하고 관리하기 위해, 저희는 **그래프 데이터 모델**이라는 '설계도'를 사용합니다. 이 설계도는 마치 건물을 짓기 전에 어떤 방을 어디에 두고, 방과 방 사이를 어떻게 연결할지 미리 계획하는 것과 같아요.

## 그래프 데이터 모델이란 무엇일까요?

그래프 데이터 모델은 정보를 **노드(Nodes)**와 **관계(Relationships)**, 그리고 그들의 **속성(Properties)**으로 표현하는 방식입니다.

*   **노드 (Node, 정점):** 정보의 개별 단위를 의미합니다. 우리 프로젝트에서는 '논문', '저자', '키워드', '학술지' 등이 노드가 될 수 있어요. 사람, 장소, 사물 등 세상의 모든 '명사'에 해당한다고 생각할 수 있습니다.
    *   예: 👩‍🔬 저자 "김연구", 📄 논문 "그래프 DB의 미래", 🔑 키워드 "인공지능"
*   **관계 (Relationship, 엣지):** 노드들 사이의 연결고리를 의미합니다. 어떤 노드가 다른 노드와 어떤 식으로 연관되어 있는지 나타내죠. '동사'와 비슷하다고 생각할 수 있어요.
    *   예: 김연구 (저자) -[작성함]-> "그래프 DB의 미래" (논문)
    *   예: "그래프 DB의 미래" (논문) -[가지고 있음]-> "인공지능" (키워드)
*   **속성 (Property):** 노드나 관계가 가질 수 있는 구체적인 정보입니다. 예를 들어, '저자' 노드는 '이름', '소속' 같은 속성을 가질 수 있고, '논문' 노드는 '제목', '출판 연도' 같은 속성을 가질 수 있습니다. 관계 또한 속성을 가질 수 있어요 (예: '참여함' 관계에 '역할' 속성).

이 세 가지 요소를 사용하면, 복잡하게 얽힌 정보들을 마치 지도처럼 명확하게 그려낼 수 있습니다. Neo4j는 바로 이러한 그래프 데이터를 저장하고 관리하는 데 특화된 데이터베이스입니다.

## 우리 프로젝트의 그래프 데이터 모델

우리 `SKN10-4th-1Team` 프로젝트에서는 논문, 저자, 키워드, 학술지 정보를 다음과 같은 구조로 Neo4j 데이터베이스에 저장하기로 설계했습니다.

```mermaid
graph LR
    A[Author: 저자] -- AUTHORED_BY (작성함) --> P[Article: 논문]
    P -- PUBLISHED_IN (게재됨) --> J[Journal: 학술지]
    P -- HAS_KEYWORD (포함함) --> K[Keyword: 키워드]

    subgraph 주요 정보 단위 (노드)
        A
        P
        J
        K
    end

    style A fill:#A6E3E9,stroke:#333,stroke-width:2px
    style P fill:#F9ED69,stroke:#333,stroke-width:2px
    style J fill:#F08A5D,stroke:#333,stroke-width:2px
    style K fill:#B2F7EF,stroke:#333,stroke-width:2px
```

위 그림에서 볼 수 있듯이,
*   **Author (저자)** 노드는 `AUTHORED_BY` 관계를 통해 **Article (논문)** 노드와 연결됩니다. (예: "홍길동" 저자가 "율도국 건설 방안" 논문을 작성했다.)
*   **Article (논문)** 노드는 `PUBLISHED_IN` 관계를 통해 **Journal (학술지)** 노드와 연결됩니다. (예: "율도국 건설 방안" 논문이 "이상사회연구" 학술지에 게재되었다.)
*   **Article (논문)** 노드는 `HAS_KEYWORD` 관계를 통해 **Keyword (키워드)** 노드와 연결됩니다. (예: "율도국 건설 방안" 논문은 "사회개혁", "리더십" 키워드를 가지고 있다.)

이런 구조를 통해 "홍길동 저자가 쓴 논문들 중 '리더십' 키워드를 가진 논문은 무엇인가?"와 같은 질문에 대한 답을 쉽게 찾을 수 있게 됩니다.

## 코드로 모델 정의하기 (`neomodel` 라이브러리)

이러한 그래프 데이터 모델을 파이썬 코드로는 어떻게 표현할까요? 저희는 `neomodel`이라는 라이브러리를 사용합니다. `neomodel`은 파이썬 클래스를 정의하는 것만으로 Neo4j 데이터베이스의 노드와 관계를 쉽게 다룰 수 있게 도와주는 도구입니다.

먼저, 우리 프로젝트가 Neo4j 데이터베이스에 연결될 수 있도록 설정하는 코드가 필요합니다. `api/models.py` 파일 상단에 다음과 같은 코드가 있습니다.

```python
# api/models.py
from neomodel import config
import os
from dotenv import load_dotenv

load_dotenv() # .env 파일에서 환경 변수 로드

# Neo4j 데이터베이스 연결 주소 설정
NEO4J_BOLT_URL = os.getenv('NEO4J_BOLT_URL', 'bolt://neo4j:your_strong_password@localhost:7687')
config.DATABASE_URL = NEO4J_BOLT_URL
print(f"Neomodel 연결 설정: {config.DATABASE_URL}")
```
이 코드는 Neo4j 데이터베이스 서버의 주소 (`NEO4J_BOLT_URL`)를 읽어와 `neomodel`이 데이터베이스와 통신할 수 있도록 설정합니다. 이 주소는 여러분의 로컬 환경이나 Docker 설정에 따라 달라질 수 있습니다.

이제 각 노드 유형을 파이썬 클래스로 정의해 보겠습니다.

### 1. 저자 (Author) 노드 정의

```python
# api/models.py (계속)
from neomodel import (StructuredNode, StringProperty, RelationshipTo, RelationshipFrom)
# ... 다른 import 구문들 ...

class Author(StructuredNode):
    full_name = StringProperty(index=True, required=True) # 저자 전체 이름
    last_name = StringProperty(index=True)                 # 성
    fore_name = StringProperty()                           # 이름
    # ... 다른 저자 정보 속성들 ...

    # 관계 정의: Author 노드가 Article 노드로 'AUTHORED_BY' 관계를 가짐
    articles = RelationshipTo('Article', 'AUTHORED_BY')
```
*   `StructuredNode`: 이 클래스를 상속받으면 Neo4j의 노드가 됩니다.
*   `StringProperty`: 문자열 데이터를 저장하는 속성입니다. `index=True`는 이 속성으로 검색을 빠르게 할 수 있도록 도와주고, `required=True`는 이 속성값이 반드시 있어야 함을 의미합니다.
*   `RelationshipTo('Article', 'AUTHORED_BY')`: `Author` 노드에서 `Article` 노드로 향하는 `AUTHORED_BY`라는 이름의 관계를 정의합니다. 즉, "저자가 논문을 작성했다"는 관계를 나타냅니다.

### 2. 논문 (Article) 노드 정의

```python
# api/models.py (계속)
from neomodel import UniqueIdProperty, IntegerProperty, DateProperty, JSONProperty

class Article(StructuredNode):
    uid = UniqueIdProperty() # neomodel이 자동으로 생성하는 고유 ID
    pmid = StringProperty(unique_index=True, required=True) # 논문 고유 식별자 (예: PubMed ID)
    title = StringProperty(required=True, max_length=500)   # 논문 제목
    abstract = StringProperty()                             # 초록
    publication_year = IntegerProperty(index=True)          # 출판 연도
    # ... 다른 논문 정보 속성들 ...

    # 임베딩 벡터 저장 (나중에 자세히 다룰 예정)
    title_embedding = JSONProperty()
    abstract_embedding = JSONProperty()
    combined_embedding = JSONProperty()

    # 관계 정의
    # Article 노드가 Author 노드로부터 'AUTHORED_BY' 관계를 받음
    authors = RelationshipFrom(Author, 'AUTHORED_BY')
    # Article 노드가 Journal 노드로 'PUBLISHED_IN' 관계를 가짐
    journal = RelationshipTo('Journal', 'PUBLISHED_IN')
    # Article 노드가 Keyword 노드로 'HAS_KEYWORD' 관계를 가짐
    keywords = RelationshipTo('Keyword', 'HAS_KEYWORD')
```
*   `UniqueIdProperty`: 각 노드마다 자동으로 고유한 ID를 부여합니다.
*   `IntegerProperty`: 정수형 데이터를 저장하는 속성입니다.
*   `JSONProperty`: JSON 형식의 데이터를 저장할 수 있는 속성입니다. 저희는 여기에 논문 제목이나 초록의 **임베딩 벡터**라는 것을 저장할 예정입니다. 임베딩 벡터는 텍스트의 의미를 숫자로 표현한 것으로, [언어 모델 연동 (GeminiLLM)](03_언어_모델_연동__geminillm__.md) 챕터와 [벡터/그래프 검색 저장소 (Neo4jVectorSearch)](04_벡터_그래프_검색_저장소__neo4jvectorsearch__.md) 챕터에서 더 자세히 다룰 것입니다.
*   `RelationshipFrom(Author, 'AUTHORED_BY')`: `Author` 노드에서 이 `Article` 노드로 오는 `AUTHORED_BY` 관계를 정의합니다. 즉, "이 논문은 어떤 저자들에 의해 작성되었다"를 나타냅니다.
*   `RelationshipTo(...)`: 다른 노드 유형 (`Journal`, `Keyword`)과의 관계도 유사하게 정의합니다.

### 3. 키워드 (Keyword) 및 학술지 (Journal) 노드 정의

```python
# api/models.py (계속)
class Keyword(StructuredNode):
    term = StringProperty(unique_index=True, required=True) # 키워드 용어
    # Keyword 노드가 Article 노드로부터 'HAS_KEYWORD' 관계를 받음
    articles = RelationshipFrom('Article', 'HAS_KEYWORD')

class Journal(StructuredNode):
    name = StringProperty(unique_index=True, required=True) # 학술지 이름
    issn = StringProperty(index=True)                      # 국제 표준 일련 번호
    # Journal 노드가 Article 노드로부터 'PUBLISHED_IN' 관계를 받음
    articles = RelationshipFrom('Article', 'PUBLISHED_IN')
```
`Keyword`와 `Journal` 노드도 `Author`, `Article`과 유사하게 필요한 속성과 관계를 정의합니다. 예를 들어 `Keyword`의 `articles` 속성은 특정 키워드를 가지고 있는 모든 논문들을 가리키게 됩니다.

이렇게 파이썬 클래스로 모델을 정의하면, 우리는 마치 일반적인 파이썬 객체를 다루듯이 Neo4j 데이터베이스의 정보를 생성하고, 읽고, 수정하고, 삭제할 수 있게 됩니다.

## 모델이 실제로 사용되는 과정 (간단히 살펴보기)

우리가 위에서 정의한 파이썬 클래스들은 `neomodel` 라이브러리를 통해 Neo4j 데이터베이스와 상호작용합니다. 예를 들어, 새로운 저자와 그 저자가 쓴 논문을 데이터베이스에 추가하고 싶다고 가정해 봅시다.

1.  파이썬 코드에서 `Author` 객체와 `Article` 객체를 생성합니다.
2.  `author.articles.connect(article_obj)` 와 같은 코드를 사용하여 두 객체 사이에 `AUTHORED_BY` 관계를 설정합니다.
3.  `author.save()` 와 `article_obj.save()` 를 호출하면, `neomodel`은 이 정보를 바탕으로 Neo4j가 이해할 수 있는 명령어(Cypher 쿼리)를 생성하여 데이터베이스로 보냅니다.
4.  Neo4j 데이터베이스는 이 명령어를 받아 실제 노드와 관계를 저장합니다.

이 과정을 간단한 순서도로 표현하면 다음과 같습니다.

```mermaid
sequenceDiagram
    participant UserCode as 파이썬 코드 (모델 사용)
    participant NeomodelLib as neomodel 라이브러리
    participant Neo4jDB as Neo4j 데이터베이스

    UserCode->>NeomodelLib: author = Author(full_name="홍길동").save()
    UserCode->>NeomodelLib: article = Article(title="새 논문").save()
    UserCode->>NeomodelLib: author.articles.connect(article)
    Note right of NeomodelLib: "홍길동" Author 노드 생성 요청
    NeomodelLib->>Neo4jDB: Cypher 쿼리 (Author 노드 생성)
    Neo4jDB-->>NeomodelLib: 생성 완료
    Note right of NeomodelLib: "새 논문" Article 노드 생성 요청
    NeomodelLib->>Neo4jDB: Cypher 쿼리 (Article 노드 생성)
    Neo4jDB-->>NeomodelLib: 생성 완료
    Note right of NeomodelLib: 두 노드 간 관계 연결 요청
    NeomodelLib->>Neo4jDB: Cypher 쿼리 (AUTHORED_BY 관계 생성)
    Neo4jDB-->>NeomodelLib: 연결 완료
    NeomodelLib-->>UserCode: 모든 작업 완료
end
```

## 스키마 설치: Neo4j에게 우리 모델 알려주기

우리가 파이썬 코드로 모델을 정의했지만, Neo4j 데이터베이스 자체도 이 구조에 대해 알고 있어야 검색 속도를 높이는 등의 최적화를 할 수 있습니다. 이를 위해 `neomodel`은 '스키마 인덱스 및 제약 조건 설치' 기능을 제공합니다.

`api/models.py` 파일 하단에는 다음과 같은 함수가 있습니다.

```python
# api/models.py (계속)
def install_neomodel_labels():
    """neomodel 모델 정의에 따른 인덱스 및 제약 조건을 Neo4j에 설치합니다."""
    print("Neomodel 레이블 및 제약 조건 설치 중...")
    from neomodel import install_labels
    try:
        install_labels(Article)
        install_labels(Author)
        install_labels(Journal)
        install_labels(Keyword)
        print("Neomodel 레이블 및 제약 조건 설치 완료.")
    except Exception as e:
        print(f"레이블 설치 중 오류 발생: {e}")
```
이 `install_neomodel_labels` 함수는 우리가 정의한 `Article`, `Author`, `Journal`, `Keyword` 클래스 정보를 바탕으로 Neo4j 데이터베이스에 "이런 종류의 노드들이 있고, 특정 속성(예: `pmid`, `full_name`)은 고유하거나 검색을 위해 인덱싱 되어야 한다"고 알려줍니다. 마치 책의 맨 뒤에 있는 '찾아보기(색인)'를 만드는 것과 같아서, 나중에 데이터를 검색할 때 훨씬 빠르게 원하는 정보를 찾을 수 있게 해줍니다.

이 함수는 보통 프로젝트 초기 설정 시 한 번 실행하거나, Django 프로젝트의 경우 `manage.py` 명령어를 통해 실행합니다. 이 부분은 나중에 [장고 프로젝트 설정 및 실행](07_장고_프로젝트_설정_및_실행_.md) 챕터에서 더 자세히 다룰 예정입니다.

## 정리하며

이번 장에서는 우리 프로젝트의 핵심 '설계도'인 **그래프 데이터 모델**에 대해 알아보았습니다.
*   정보를 **노드**(점)와 **관계**(선)로 표현하는 그래프의 기본 개념을 이해했습니다.
*   우리 프로젝트에서는 `Author`, `Article`, `Journal`, `Keyword`라는 주요 노드들과 `AUTHORED_BY`, `PUBLISHED_IN`, `HAS_KEYWORD`라는 관계를 사용하여 연구 정보를 구조화합니다.
*   파이썬 `neomodel` 라이브러리를 사용하여 이러한 모델을 코드로 정의하는 방법을 살펴보았습니다.
*   이 모델이 어떻게 복잡한 연구 정보 네트워크를 표현하고, 특정 질문에 대한 답을 찾는 데 도움을 주는지 감을 잡으셨기를 바랍니다.

이 탄탄한 설계도를 바탕으로, 이제 실제 데이터를 이 모델에 채워 넣을 준비를 해야 합니다. 다음 장에서는 이 모델에 맞는 데이터를 어떻게 수집하고 가공하는지에 대해 알아보겠습니다.

바로 [제2장: 데이터 수집 및 전처리 (스크립트)](02_데이터_수집_및_전처리__스크립트__.md)에서 만나요!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)