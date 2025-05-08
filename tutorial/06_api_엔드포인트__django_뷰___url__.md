# Chapter 6: API 엔드포인트 (Django 뷰 & URL)


안녕하세요! 이전 [제5장: RAG 파이프라인 (HybridGraphFlow)](05_rag_파이프라인__hybridgraphflow__.md)에서는 사용자의 복잡한 질문에도 척척 답변을 찾아내는 전문가 팀, `HybridGraphFlow`에 대해 배웠습니다. 이렇게 강력한 질문 해결사를 만들었으니, 이제 사용자들이 실제로 이 시스템을 어떻게 사용할 수 있을지 알아볼 차례입니다. 예를 들어, 웹 브라우저의 챗봇 화면에서 사용자가 질문을 입력하고 '전송' 버튼을 누르면, 그 질문이 우리 전문가 팀에게 전달되고, 생성된 답변이 다시 화면에 나타나야겠죠?

이번 장에서는 바로 이 '소통의 다리' 역할을 하는 **API 엔드포인트**에 대해 배울 거예요. API 엔드포인트는 웹 애플리케이션(챗봇 화면)이 우리 프로젝트의 백엔드 시스템(전문가 팀 `HybridGraphFlow`)과 대화할 수 있도록 만들어주는 '창구'와 같습니다. 이 창구를 통해 사용자의 요청이 들어오고, 처리된 결과가 다시 사용자에게 전달된답니다. 우리는 파이썬 웹 프레임워크인 **Django(장고)**를 사용하여 이 창구를 만들 거예요.

## API 엔드포인트란 무엇일까요? (우체국 비유 📮)

API 엔드포인트를 이해하기 위해 우체국을 한번 떠올려 볼까요? 우리가 편지를 보내거나 소포를 찾으려면 우체국의 특정 창구로 가야 하죠. 각 창구는 특정 업무를 담당합니다.

*   **URL (Uniform Resource Locator, 주소)**: 웹 세상의 '주소'입니다. 우체국의 '1번 창구: 소포 접수', '2번 창구: 우표 구매'처럼, 각 URL은 특정 기능을 수행하는 고유한 경로를 나타냅니다. 예를 들어, 우리 챗봇 시스템에서는 `우리챗봇.com/질문하기` 같은 주소가 있을 수 있겠죠.
*   **뷰 (View Function, 창구 직원)**: 특정 URL(창구)로 요청이 들어왔을 때, 그 요청을 실제로 처리하는 '창구 직원'입니다. 사용자의 질문을 받아서 `HybridGraphFlow` 전문가 팀에게 전달하고, 받은 답변을 다시 사용자에게 보내주는 역할을 합니다. Django에서는 파이썬 함수 형태로 만들어집니다.
*   **Django (장고, 우체국 시스템 전체)**: 이 모든 URL과 뷰 함수를 연결하고 관리하는 '우체국 시스템' 그 자체입니다. 사용자가 특정 URL로 접속하면, Django는 해당 URL에 연결된 뷰 함수를 찾아서 실행시켜 줍니다.

즉, 사용자가 챗봇에 질문을 입력하면:
1.  챗봇 화면은 특정 URL(예: `/api/search/`)로 "이 질문 좀 처리해주세요!" 하고 요청을 보냅니다.
2.  Django는 이 URL을 보고, "아, 이 주소는 `search_view`라는 직원이 담당이지!" 하고 해당 뷰 함수를 호출합니다.
3.  `search_view` 함수(창구 직원)는 질문을 받아 [제5장: RAG 파이프라인 (HybridGraphFlow)](05_rag_파이프라인__hybridgraphflow__.md)에게 전달하여 답변을 얻어냅니다.
4.  얻어낸 답변을 다시 챗봇 화면으로 보내주면, 사용자는 화면에서 답변을 볼 수 있게 됩니다.

이 전체 과정을 그림으로 보면 다음과 같습니다:

```mermaid
graph LR
    A[사용자 (챗봇 화면)] -- 1. 질문 전송 (HTTP 요청) --> B(웹 서버);
    B -- 2. Django URL 라우팅 --> C[URL 패턴 일치 확인 (urls.py)];
    C -- 3. 해당 뷰 함수 호출 --> D[뷰 함수 실행 (views.py)];
    D -- 4. RAG 파이프라인 호출 --> E([HybridGraphFlow 전문가팀]);
    E -- 5. 답변 생성 --> D;
    D -- 6. 응답 준비 (JSON) --> B;
    B -- 7. 답변 전달 (HTTP 응답) --> A;

    style A fill:#A6E3E9,stroke:#333,stroke-width:2px
    style B fill:#F9ED69,stroke:#333,stroke-width:2px
    style C fill:#F08A5D,stroke:#333,stroke-width:2px
    style D fill:#B2F7EF,stroke:#333,stroke-width:2px
    style E fill:#FFB6C1,stroke:#333,stroke-width:2px
```

## 우리 프로젝트의 창구 만들기: `urls.py`와 `views.py`

이제 Django를 사용하여 이 '창구'들을 어떻게 만드는지 살펴보겠습니다. 주로 `api`라는 앱(Django 프로젝트 내의 작은 기능 단위) 폴더 안에 있는 `urls.py`와 `views.py` 두 파일이 핵심적인 역할을 합니다.

### 1. `api/urls.py`: 어떤 주소로 가면 누가 있나요? (주소 안내판)

`api/urls.py` 파일은 어떤 URL 경로로 요청이 왔을 때 어떤 뷰 함수를 실행할지 연결해주는 '주소 안내판' 역할을 합니다.

```python
# api/urls.py
from django.urls import path
from . import views # 같은 api 폴더 안에 있는 views.py 파일을 가져옵니다

app_name = 'api' # 이 URL들의 그룹 이름을 'api'로 정합니다 (나중에 편리해요!)

urlpatterns = [
    # 웹 브라우저에서 '우리사이트주소/api/search/'로 접속하면,
    # views.py 파일 안에 있는 search_view 함수를 실행하라는 뜻입니다.
    # name='search_path'는 이 경로에 'search_path'라는 별명을 붙여줍니다.
    path('search/', views.search_view, name='search_path'),

    # '우리사이트주소/api/document_info/'로 접속하면,
    # views.py 파일 안의 document_info_view 함수를 실행합니다.
    path('document_info/', views.document_info_view, name='document_info'),
]
```

*   `from . import views`: 현재 폴더(api)에 있는 `views.py` 파일을 사용하겠다고 알려줍니다.
*   `urlpatterns`: URL 패턴과 뷰 함수를 짝지어 놓은 목록입니다.
*   `path('search/', views.search_view, name='search_path')`:
    *   `'search/'`: 사용자가 접속할 URL 경로입니다. (실제 전체 주소는 `http://우리서버주소/api/search/` 와 같이 됩니다.)
    *   `views.search_view`: 이 경로로 요청이 오면 `views.py` 파일에 있는 `search_view` 라는 함수를 실행하라는 의미입니다.
    *   `name='search_path'`: 이 URL 경로에 'search_path'라는 별명을 붙여서 나중에 코드 다른 곳에서 이 별명으로 URL을 쉽게 참조할 수 있게 합니다.

이 파일을 통해 우리는 "챗봇 질문 처리는 `/api/search/` 창구에서 `search_view` 직원이 담당하고, 특정 논문 상세 정보 요청은 `/api/document_info/` 창구에서 `document_info_view` 직원이 담당한다"고 Django에게 알려주는 것입니다.

### 2. `api/views.py`: 실제 일을 하는 창구 직원들!

`api/views.py` 파일에는 `urls.py`에서 지정한 뷰 함수들이 실제로 정의되어 있습니다. 이 함수들이야말로 요청을 받아 처리하고 응답을 만들어내는 '창구 직원'들이죠.

먼저, 우리 프로젝트의 전문가 팀(`HybridGraphFlow`)과 슈퍼 사서(`Neo4jVectorSearch`)를 준비시킵니다.

```python
# api/views.py
from django.shortcuts import render # HTML 파일을 화면에 보여줄 때 사용
from django.http import HttpRequest, JsonResponse # 요청/응답 객체, JSON 응답
import json # JSON 데이터를 다룰 때 사용

# 5장에서 만든 RAG 전문가팀 HybridGraphFlow를 가져옵니다.
from rag_pipeline.graph_flow import HybridGraphFlow
# 4장에서 만든 슈퍼 사서 Neo4jVectorSearch도 가져옵니다.
from rag_pipeline.vector_store import Neo4jVectorSearch

# 애플리케이션이 시작될 때 전문가팀과 슈퍼 사서를 미리 한 번만 만들어 둡니다.
# 이렇게 하면 매번 요청이 올 때마다 새로 만들 필요가 없어 효율적입니다.
graph_rag = HybridGraphFlow()
vector_search = Neo4jVectorSearch()
```
이렇게 `graph_rag`와 `vector_search` 객체를 파일 최상단에 만들어두면, 장고 애플리케이션이 실행될 때 딱 한 번만 생성되어 모든 요청 처리 시 공유해서 사용할 수 있습니다. 마치 우체국 문 열 때 직원들이 이미 자리에 앉아있는 것과 같죠!

#### 챗봇 질문 처리 창구: `search_view` 함수

이 함수는 사용자가 챗봇에 질문을 입력했을 때 그 요청을 받아 처리하고, `HybridGraphFlow`를 통해 얻은 답변을 다시 챗봇 화면으로 보내줍니다.

```python
# api/views.py (계속)
def search_view(request: HttpRequest): # 사용자의 요청(HttpRequest)을 받습니다.
    """
    검색 페이지를 보여주거나 (GET 요청),
    사용자의 질문을 받아 RAG 파이프라인을 호출하고 결과를 반환합니다 (POST 요청).
    """
    if request.method == 'POST': # 사용자가 질문을 '보내기' (POST 방식) 했을 때
        try:
            # 사용자가 보낸 데이터(질문, 대화 기록 등)를 JSON 형태로 받습니다.
            data = json.loads(request.body)
            query = data.get('query', '') # 'query' 라는 이름으로 온 질문 내용
            chat_history = data.get('chat_history', []) # 이전 대화 내용

            if query: # 질문 내용이 있다면
                # ✨ 핵심! 전문가팀(graph_rag)에게 질문과 대화 기록을 넘겨 답변을 받습니다.
                result = graph_rag.query(query, chat_history)
                
                # 추가로, 첫 번째 검색된 문서의 연결 관계도 가져와 볼까요? (선택 사항)
                connections = []
                if result.get('retrieved_docs') and len(result['retrieved_docs']) > 0:
                    first_doc_pmid = result['retrieved_docs'][0].get('pmid')
                    if first_doc_pmid:
                        connections = vector_search.find_article_connections(first_doc_pmid)
                
                # 최종 응답 데이터 구성
                response_data = {
                    'answer': result.get('answer'), # AI 답변
                    'messages': result.get('messages'), # 전체 대화 기록
                    'retrieved_docs': result.get('retrieved_docs'), # 참고 문서
                    'related_info': result.get('related_info'), # 관련 노드 정보
                    'connections': connections, # 위에서 가져온 연결 관계
                    'query_type': result.get('query_type'), # 분석된 질문 유형
                    'graph_context': result.get('graph_context', {}), # 그래프 문맥
                    'citations': result.get('citations', []), # 인용 정보
                }
                return JsonResponse(response_data) # 결과를 JSON 형태로 화면에 돌려줍니다.
            else:
                return JsonResponse({'error': '검색어를 입력해주세요.'}, status=400)
        
        except Exception as e: # 오류가 발생하면
            print(f"search_view 오류: {e}")
            return JsonResponse({'error': f'죄송합니다, 처리 중 오류가 발생했습니다: {str(e)}'}, status=500)
    
    # 사용자가 처음 페이지에 접속했을 때 (GET 방식)
    # 'api/chatbot.html' 파일을 화면에 보여줍니다.
    return render(request, 'api/chatbot.html', {})
```
*   `request: HttpRequest`: Django가 전달해주는 사용자의 요청 정보 꾸러미입니다.
*   `request.method == 'POST'`: 사용자가 데이터를 제출했는지 (예: 챗봇 메시지 전송) 확인합니다. 'GET'은 보통 페이지를 처음 불러올 때 사용됩니다.
*   `json.loads(request.body)`: 웹 브라우저(챗봇 화면)에서 보낸 데이터는 보통 JSON이라는 약속된 형식으로 오는데, 이걸 파이썬이 알아볼 수 있게 변환합니다.
*   `query = data.get('query', '')`: 변환된 데이터에서 'query'라는 이름으로 담겨 온 실제 사용자 질문을 꺼냅니다.
*   `result = graph_rag.query(query, chat_history)`: 여기가 핵심! [제5장: RAG 파이프라인 (HybridGraphFlow)](05_rag_파이프라인__hybridgraphflow__.md)에서 만든 `graph_rag` 객체의 `query` 메서드를 호출하여 답변을 생성합니다.
*   `JsonResponse(response_data)`: `graph_rag`가 돌려준 결과(`result`)와 추가 정보(`connections`)를 다시 JSON 형태로 포장해서 웹 브라우저로 돌려줍니다. 웹 브라우저의 자바스크립트 코드가 이 JSON 데이터를 받아서 화면에 예쁘게 표시해 줄 거예요.
*   `render(request, 'api/chatbot.html', {})`: 만약 `POST` 요청이 아니라 `GET` 요청(예: 사용자가 처음 챗봇 페이지에 접속)이면, `chatbot.html`이라는 HTML 파일을 화면에 보여줍니다. (이 HTML 파일은 이 튜토리얼 범위 밖이지만, 실제 챗봇의 사용자 인터페이스를 담고 있습니다.)

#### 특정 논문 상세 정보 창구: `document_info_view` 함수

사용자가 챗봇 결과에서 특정 논문을 클릭했을 때, 그 논문의 저자, 키워드, 관련 논문 등 더 자세한 정보를 보여주는 창구입니다.

```python
# api/views.py (계속)
def document_info_view(request: HttpRequest):
    """
    사용자가 선택한 문서(PMID 기준)의 상세 정보와 연결 관계를 제공합니다.
    """
    if request.method == 'GET': # 보통 URL에 pmid를 포함시켜 정보를 요청합니다.
        try:
            pmid = request.GET.get('pmid') # URL에서 'pmid' 값을 가져옵니다.
                                           # 예: /api/document_info/?pmid=12345
            if not pmid:
                return JsonResponse({'error': 'PMID가 제공되지 않았습니다.'}, status=400)

            # ✨ 슈퍼 사서(vector_search)에게 논문 ID를 주고 관련 정보를 받아옵니다.
            related_info = vector_search.get_related_nodes(pmid)
            # 해당 논문과 연결된 다른 논문 정보도 가져옵니다.
            connections = vector_search.find_article_connections(pmid)
            
            # 응답 데이터 구성
            response_data = {
                'related_info': related_info,
                'connections': connections
            }
            return JsonResponse(response_data) # 결과를 JSON 형태로 돌려줍니다.

        except Exception as e:
            print(f"document_info_view 오류: {e}")
            return JsonResponse({'error': f'문서 정보를 가져오는 중 오류 발생: {str(e)}'}, status=500)
    
    return JsonResponse({'error': '잘못된 요청 방식입니다.'}, status=405) # GET 방식만 허용
```
*   `request.GET.get('pmid')`: URL 주소 뒤에 `?pmid=12345` 와 같이 붙어서 오는 파라미터 값을 가져옵니다. 이 `pmid`는 논문의 고유 ID입니다.
*   `vector_search.get_related_nodes(pmid)` 와 `vector_search.find_article_connections(pmid)`: [제4장: 벡터/그래프 검색 저장소 (Neo4jVectorSearch)](04_벡터_그래프_검색_저장소__neo4jvectorsearch__.md)의 기능을 사용하여 특정 논문의 상세 정보(저자, 키워드 등)와 그 논문과 연결된 다른 논문 정보를 가져옵니다.
*   마찬가지로 `JsonResponse`를 통해 결과를 웹 브라우저로 전달합니다.

## 요청 처리 과정 한눈에 보기 🕵️‍♂️

사용자가 챗봇에 질문을 하면 어떤 일이 일어나는지 순서도로 다시 한번 정리해볼까요?

```mermaid
sequenceDiagram
    participant UserClient as 사용자 (챗봇 화면)
    participant DjangoServer as Django 웹 서버
    participant URLs as api/urls.py (주소 안내판)
    participant Views as api/views.py (search_view 직원)
    participant RAGFlow as HybridGraphFlow (전문가팀)

    UserClient->>DjangoServer: POST /api/search/ (질문: "코로나 백신 연구?")
    DjangoServer->>URLs: "/api/search/" 주소에 맞는 직원 찾아줘!
    URLs-->>DjangoServer: views.search_view 직원이 담당합니다!
    DjangoServer->>Views: search_view 직원, 이 요청(질문) 처리해주세요!
    Views->>RAGFlow: 전문가팀! "코로나 백신 연구?" 질문에 대한 답변 부탁해요!
    RAGFlow-->>Views: (처리 후) 여기 답변과 관련 자료입니다!
    Views-->>DjangoServer: (JSON으로 포장해서) 처리 결과입니다!
    DjangoServer-->>UserClient: (화면에 표시할 수 있게) JSON 응답 전달!
end
```
이처럼 `urls.py`와 `views.py`는 Django라는 잘 짜인 시스템 안에서 외부 요청을 받아 우리 프로젝트 내부의 핵심 로직(`HybridGraphFlow`, `Neo4jVectorSearch`)과 연결해주는 매우 중요한 역할을 합니다.

## 정리하며 🌟

이번 장에서는 웹 애플리케이션이 우리 RAG 시스템과 통신할 수 있도록 하는 '창구', 즉 **API 엔드포인트**를 Django를 이용해 만드는 방법을 배웠습니다.

*   **URL**은 특정 기능을 요청하는 '주소'와 같고, **뷰 함수**는 그 요청을 처리하는 '창구 직원'과 같다는 것을 이해했습니다.
*   `api/urls.py` 파일이 URL과 뷰 함수를 연결하는 '주소 안내판' 역할을 한다는 것을 보았습니다.
*   `api/views.py` 파일에 있는 뷰 함수들이 실제로 요청을 받아, [제5장: RAG 파이프라인 (HybridGraphFlow)](05_rag_파이프라인__hybridgraphflow__.md)나 [제4장: 벡터/그래프 검색 저장소 (Neo4jVectorSearch)](04_벡터_그래프_검색_저장소__neo4jvectorsearch__.md) 같은 핵심 모듈을 호출하고, 그 결과를 **JSON** 형태로 다시 사용자(웹 브라우저)에게 전달하는 과정을 살펴보았습니다.
*   `search_view` (챗봇 질문 처리)와 `document_info_view` (특정 문서 상세 정보)라는 두 가지 주요 API 엔드포인트 예시를 통해 실제 코드 작동 방식을 이해했습니다.

이제 우리는 사용자와 상호작용할 수 있는 '창구'까지 모두 만들었습니다! 데이터 모델 설계부터 데이터 수집, LLM 연동, 검색 기능, RAG 파이프라인, 그리고 이 모든 것을 외부와 연결하는 API 엔드포인트까지, 우리 프로젝트의 모든 핵심 구성 요소들을 다 살펴보았네요.

다음 장에서는 드디어 이 모든 것을 하나로 합쳐 실제 Django 프로젝트를 설정하고 실행하여, 우리가 만든 챗봇 시스템이 동작하는 모습을 직접 확인해보는 시간을 갖겠습니다!

바로 [제7장: 장고 프로젝트 설정 및 실행](07_장고_프로젝트_설정_및_실행_.md)에서 만나요!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)