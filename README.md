# SK Networks AI CAMP 10기 - 4th 1Team: MediSearch AI

- **주제:** 알츠하이머 연구를 위한 의학 논문 검색 시스템
- **개발 기간:** 2025.05.09 ~ 2025.05.12

---

## 📌 목차

- [01. 팀 소개](#01-팀-소개)
- [02. 프로젝트 개요](#02-프로젝트-개요)
- [03. 기술 스택](#03-기술-스택)
- [04. 시스템 아키텍처](#04-시스템-아키텍처)
- [05. 디렉토리 구조](#05-디렉토리-구조)
- [06. 팀 구성 및 역할](#06-팀-구성-및-역할)
- [07. 주요 기능](#07-주요-기능)
- [08. 설치 및 실행 가이드](#08-설치-및-실행-가이드)
- [09. 향후 개선 계획](#09-향후-개선-계획)

---

## 1. 팀 소개

### ✅ 팀명: MediSearch AI
알츠하이머 연구를 위한 최첨단.맞춤형 의학 논문 검색 도구

### ✅ 팀원 소개

| 황인호 | 경규휘 | 권석현 | 이종원 | 장윤홍 |
|--------|--------|--------|--------|--------|
| 기술 설계 | RAG 개발 | 백엔드 개발 | 프론트엔드 개발 | 데이터 처리 |

---

## 2. 프로젝트 개요

### ✅ 프로젝트 명: MediSearch AI
### ✅ 프로젝트 소개: 알츠하이머 연구를 위한 의학 논문 검색 시스템

MediSearch AI는 알츠하이머 관련 의학 논문 검색을 위한 고급 RAG(Retrieval-Augmented Generation) 시스템입니다. 이 시스템은 의학 연구자들이 알츠하이머 논문을 효과적으로 검색하고 관련 정보를 얻을 수 있도록 설계되었습니다.

### ✅ 프로젝트 주요 기능

- **하이브리드 검색 시스템**: 벡터 검색, 그래프 검색, 하이브리드 검색 모드 지원
- **논문 관계 분석**: 저자, 키워드, 주제 간의 연결 관계 시각화
- **LLM 기반 응답 생성**: Google Gemini 모델을 활용한 정확한 정보 요약
- **다국어 지원**: 한국어 질의에 대한 한국어 응답 생성

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

![RAG 시스템 아키텍처](https://i.imgur.com/7FxRUwy.png)

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
├── user/                # 로그인, 회원가입 구현
├── requirements.txt     # 의존성 패키지 목록
└── manage.py            # Django 관리 스크립트
```

## 6. 팀 구성 및 역할

### ✅ 업무 분담표

| 담당 영역 | 주요 업무 | 세부 작업 | 작업자 |
|---------|---------|----------|----------|
| **RAG 파이프라인 개발** | 검색 파이프라인 구축 | - LangGraph 워크플로우 구현<br>- 질의 분석 로직 개발<br>- 검색 전략 최적화 | 황인호 |
| **Neo4j 데이터베이스** | 그래프 DB 구축 | - 스키마 설계<br>- 데이터 적재<br>- 벡터 검색 구현<br>- 그래프 쿼리 최적화 | 장윤홍 |
| **AI 모델 통합** | LLM 통합 및 튜닝 | - Gemini 모델 통합<br>- 프롬프트 엔지니어링<br>- 응답 생성 최적화 | 이종원 |
| **백엔드 개발** | Django 서버 구현 | - API 엔드포인트 개발<br>- 사용자 인증 시스템<br>- 데이터 캐싱 | 권석현 |
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
- 검색 히스토리 저장
- 개인화된 검색 환경 제공

### ✅ 화면 구성

![홈페이지 화면](https://i.imgur.com/8FxRUwy.png)
![AI 검색 화면](https://i.imgur.com/9FxRUwy.png)

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

## 9. 향후 개선 계획

### ✅ 기능 개선 계획

- **실시간 데이터 업데이트**: 최신 PubMed 논문 자동 추가 시스템
- **개인화 검색**: 사용자 관심사 기반 맞춤형 결과 제공
- **그래프 시각화 개선**: 논문 간 연결 관계 시각화 인터페이스 고도화
- **모바일 최적화**: 모바일 환경에서의 사용성 개선

### ✅ 기술 개선 계획

- **Neo4j 벡터 인덱스 최적화**: 검색 속도 및 정확도 향상
- **LangGraph 워크플로우 고도화**: 더 복잡한 검색 패턴 지원
- **다양한 모델 지원**: 추가 LLM 모델 통합 (Claude, Llama 등)
- **다국어 지원 확장**: 영어, 한국어 외 추가 언어 지원
