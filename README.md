# SK Networks AI CAMP 10기 - 4th 1Team: MediSearch AI

- **주제:** 알츠하이머 연구를 위한 의학 논문 검색 시스템
- **개발 기간:** 2025.05.09 ~ 2025.05.12

## 📌 목차

- [1. 팀 소개](#1-팀-소개)
- [2. 프로젝트 개요](#2-프로젝트-개요)
- [3. 기술 스택](#3-기술-스택)
- [4. 시스템 아키텍처](#4-시스템-아키텍처)
- [5. 디렉토리 구조](#5-디렉토리-구조)
- [6. 팀 구성 및 역할](#6-팀-구성-및-역할)
- [7. 주요 기능](#7-주요-기능)
- [8. 설치 및 실행 가이드](#8-설치-및-실행-가이드)
- [9. 한 줄 회고](#9-한-줄-회고)

## 1. 팀 소개

### ✅ 팀명: MediSearch AI
알츠하이머 연구를 위한 최첨단.맞춤형 의학 논문 검색 도구

### ✅ 팀원 소개

| 황인호 | 경규휘 | 권석현 | 이종원 | 장윤홍 |
|--------|--------|--------|--------|--------|
|![1](https://i.namu.wiki/i/BOFPH8CsaM0suQ8bAwqQnSG1oZA3uzqmumxC6EOPnO1Y2jVraSnYFk4_02WdTMIiqzyXPmlca6ubpjuC76lwSDskcZ4Rb5WuIEZFcTvWw5szzMfIc0Kc7czfaYjQ2l1E2vlAk_eBkuBPUDrf6Wj8QA.webp)|![2](https://i.namu.wiki/i/Em1fmJtX3WTg4ATja4AbmudnC9N-y3kmaXdSw0qObITjYriIhSmuA8lIupUDgAZuhzgN-J8W3npi-8WQJu9-48eEh_Lvwl_mn0e4BXcMRDjypXxbuQ-WZuVdp94KH__lAc7zJopu3qlfv5Ftt61NxQ.webp)|![3](https://i.namu.wiki/i/qzrnG62_SvgqNMlELLr8zihfaylbiKrM4aVl15zPXuYYXlGgUhPtQ2S6KeylnXtENeKVpSoXNe5FOft4vb-9d8f6dK4KMJAlLpFou3Y0qv0lzxJqaVwG7s7Ld97VKUe8HUSXQo_aUuDXotZEE3eaQQ.webp) |![4](https://i.namu.wiki/i/DOwF6eRC1e7PaUlayI_nCI3Jxv3a4kuKNFk-S9VhpboFYFHrLLwh0__T5fPWTXDud_WsD3vnyMFJfAUnmZ3hVSuPqpoHVrhr6qvbIk28IE0J7Skjig6jXGowTyIvQEzX0u1IvIBZ1Wa9v_mzFKMjJg.webp)|![5](https://i.namu.wiki/i/q-wkyXm5gShTOpkJ3Rzag5qm3Y-3CpZtaGXTf4j8xmHgLBeMDO-hUHY5aL9sMnkFjb5k2DnUnKgAiST29Mdi4e-1CR_kpf2tlQ62WZlvusUBWWZR7iK1WU4DgTFN9W40im9lslS3pr-fsUvCY-roAw.webp)|

---

## 2. 프로젝트 개요

### ✅ 프로젝트 명: MediSearch AI
### ✅ 프로젝트 소개: 알츠하이머 연구를 위한 의학 논문 검색 시스템

MediSearch AI는 알츠하이머 관련 의학 논문 검색을 위한 고급 RAG(Retrieval-Augmented Generation) 시스템입니다. 이 시스템은 의학 연구자들이 알츠하이머 논문을 효과적으로 검색하고 관련 정보를 얻을 수 있도록 설계되었습니다.

### ✅ 프로젝트 주요 기능

- **하이브리드 검색 시스템**: 벡터 검색, 그래프 검색, 하이브리드 검색 모드 지원
- **논문 관계 분석**: 저자, 키워드, 주제 간의 연결 관계 시각화
- **LLM 기반 응답 생성**: Google Gemini 모델을 활용한 정확한 정보 요약

## 3. 기술 스택

| 카테고리 | 기술 스택 |
|----------|-------------------------------------------|
| **웹 프레임워크** | ![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white) |
| **데이터베이스** | ![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?style=for-the-badge&logo=Neo4j&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=SQLite&logoColor=white) |
| **RAG 파이프라인** | ![LangChain](https://img.shields.io/badge/LangChain-005F73?style=for-the-badge&logo=Chainlink&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-00A86B?style=for-the-badge&logo=Chainlink&logoColor=white) |
| **AI 모델** | ![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=Google&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white) |
| **프론트엔드** | ![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-38B2AC?style=for-the-badge&logo=Tailwind-CSS&logoColor=white) ![Bootstrap Icons](https://img.shields.io/badge/Bootstrap%20Icons-7952B3?style=for-the-badge&logo=Bootstrap&logoColor=white) |
| **협업 도구** | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white) ![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=Discord&logoColor=white) |

## 4. 시스템 아키텍처

### ✅ 검색 파이프라인

```
사용자 질의 → 질의 분석 → 검색 전략 선택 (벡터/그래프/하이브리드) → 관련 논문 검색 → 
→ 그래프 컨텍스트 수집 → LLM 응답 생성 → 사용자 인터페이스
```

### ✅ 데이터 구조

- **노드 유형**: 논문(Article), 저자(Author), 키워드(Keyword), 주제(Topic)
- **관계 유형**: AUTHORED_BY, HAS_KEYWORD, BELONGS_TO_TOPIC, CITES 등

### ✅ RAG 파이프라인 구성

```mermaid
sequenceDiagram
    participant 사용자
    participant 파이프라인 as HybridGraphFlow
    participant LLM_쿼리분석 as LLM (쿼리 분석)
    participant 검색모듈 as 정보 검색 (벡터/그래프 DB)
    participant LLM_답변생성 as LLM (답변 생성)

    사용자->>파이프라인: "비만과 고혈압 연관성은?"
    activate 파이프라인
    파이프라인->>LLM_쿼리분석: 이 질문은 어떤 검색이 좋을까?
    activate LLM_쿼리분석
    LLM_쿼리분석-->>파이프라인: "의미 검색과 관계 검색 둘 다 필요! (하이브리드)"
    deactivate LLM_쿼리분석
    파이프라인->>검색모듈: "비만", "고혈압" 관련 논문 검색 및 관계 정보 탐색!
    activate 검색모듈
    검색모듈-->>파이프라인: 관련 논문 목록과 주요 연결 정보 전달
    deactivate 검색모듈
    파이프라인->>LLM_답변생성: 이 정보들로 답변 만들어줘!
    activate LLM_답변생성
    LLM_답변생성-->>파이프라인: "비만과 고혈압은 밀접한 연관이 있습니다..." (생성된 답변)
    deactivate LLM_답변생성
    파이프라인-->>사용자: 답변 전달
    deactivate 파이프라인
```

- **HybridGraphFlow**: LangGraph 기반 워크플로우 관리 및 검색 전략 결정
- **Neo4jVectorSearch**: 벡터 검색 및 그래프 기반 리랭킹 수행
- **GeminiLLM**: 검색 결과 기반 응답 생성

## 5. 디렉토리 구조

```
├── api/                 # Django API 구현
├── config/              # Django 설정 파일
├── rag_pipeline/        # RAG 파이프라인 구현
│   ├── graph_flow.py    # LangGraph 워크플로우
│   ├── vector_store.py  # Neo4j 벡터 검색
│   ├── llm.py           # Gemini LLM 클라이언트
│   └── tavily_search.py # 외부 검색 통합
├── scripts/             # 유틸리티 스크립트
├── templates/           # HTML 템플릿
│   ├── api/             # 메인 앱 템플릿
│   └── user/            # 사용자 인증 템플릿
├── tutorial/            # 사용자 매뉴얼
├── user/                # 로그인, 회원가입 구현
├── requirements.txt     # 의존성 패키지 목록
└── manage.py            # Django 관리 스크립트
```

## 6. 팀 구성 및 역할

### ✅ 업무 분담표

| 담당 영역 | 주요 업무 | 세부 작업 | 작업자 |
|---------|---------|----------|----------|
| **RAG 파이프라인 개발** | 검색 파이프라인 구축 | - LangGraph 워크플로우 구현<br>- 질의 분석 로직 개발<br>- 검색 전략 최적화 | 황인호 |
| **Neo4j 데이터베이스** | 그래프 DB 구축 | - 스키마 설계<br>- 데이터 적재<br>- 벡터 검색 구현<br>- 그래프 쿼리 최적화 | 권석현 |
| **AI 모델 통합** | LLM 통합 및 튜닝 | - Gemini 모델 통합<br>- 프롬프트 엔지니어링<br>- 응답 생성 최적화 | 이종원 |
| **백엔드 개발** | Django 서버 구현 | - API 엔드포인트 개발<br>- 사용자 인증 시스템<br>- 데이터 캐싱 | 장윤홍 |
| **프론트엔드 개발** | UI/UX 구현 | - 반응형 웹 디자인<br>- 채팅 인터페이스 구현<br>- 결과 시각화 | 경규휘 |

## 7. 주요 기능

### ✅ 홈페이지

- 서비스 소개 및 주요 기능 안내
- 알츠하이머 연구 관련 정보 제공
- 네비게이션 바를 통한 직관적인 페이지 이동

### ✅ AI 검색 페이지

- 채팅 인터페이스로 질의 입력
- 검색 모드 표시 (벡터, 그래프, 하이브리드)
- 검색 결과 및 관련 논문 표시
- 논문 간 연결 관계 시각화

### ✅ 로그인/회원가입

- 사용자 인증 시스템
- 개인화된 검색 환경 제공

### ✅ 화면 구성

- **홈페이지 화면**
  
<img src="images/homepage.png" alt="홈페이지" width="700"/>

- **AI 검색 화면**

<img src="images/chatbot.png" alt="챗봇페이지" width="700"/>

## 8. 설치 및 실행 가이드

### ✅ 요구사항

- Python 3.9 이상
- Neo4j 데이터베이스 (5.x 권장)
- API 키: OpenAI, Google Gemini

### ✅ 설치 및 설정

1. 저장소 복제
```bash
git clone [repository-url]
cd SKN10-4th-1Team
```

2. 가상 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. 패키지 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정 (.env 파일)
```
NEO4J_BOLT_URL=bolt://username:password@host:port
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
SECRET_KEY=your_django_secret_key
```

5. 데이터베이스 마이그레이션
```bash
python manage.py migrate
```

6. 서버 실행
```bash
python manage.py runserver
```

7. 웹 브라우저에서 접속
```
http://localhost:8000
```

## 9. 한 줄 회고
- **황인호** : 
- **경규휘** : 
- **권석현** :
- **이종원** : 
- **장윤홍** : django 로그인 기능을 구현하면서 세션 방식 로그인, jwt 방식 로그인에 대한 이해를 얻었습니다. 그리고 팀장님이 구현한 그래프 DB가 어떤 방식으로 데이터를 저장하고 LLM이 그래프DB를 참조하는지를 배울 수 있었습니다.



