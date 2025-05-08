from django.db import models

from neomodel import config
import os
from dotenv import load_dotenv

load_dotenv() # 프로젝트 루트의.env 파일 로드

# Docker 실행 시 설정한 NEO4J_AUTH와 일치하는 URL 사용
# 형식: bolt://<username>:<password>@<host>:<port>
NEO4J_BOLT_URL = os.getenv('NEO4J_BOLT_URL', 'bolt://neo4j:your_strong_password@localhost:7687')
config.DATABASE_URL = NEO4J_BOLT_URL
# config.ENCRYPTED_CONNECTION = False # 로컬 개발 시 암호화 비활성화 (필요 시)
config.MAX_CONNECTION_POOL_SIZE = 50 # 연결 풀 크기 설정 (선택 사항)

print(f"Neomodel 연결 설정: {config.DATABASE_URL}")


# api/models.py (계속)
from neomodel import (StructuredNode, StringProperty, IntegerProperty,
                        DateProperty, ArrayProperty, UniqueIdProperty,
                        RelationshipTo, RelationshipFrom, Relationship,
                        StructuredRel, db, JSONProperty) # JSONProperty 추가
from datetime import date

# 관계 속성 정의 (예시: 저자의 순서)
# class Authorship(StructuredRel):
#     order = IntegerProperty()

class Journal(StructuredNode):
    name = StringProperty(unique_index=True, required=True)
    issn = StringProperty(index=True)
    articles = RelationshipFrom('Article', 'PUBLISHED_IN')

class Keyword(StructuredNode):
    term = StringProperty(unique_index=True, required=True) # 키워드는 고유하다고 가정
    articles = RelationshipFrom('Article', 'HAS_KEYWORD')

class Author(StructuredNode):
    # 저자 이름은 동명이인이 많으므로 unique_index는 부적합할 수 있음
    # 고유 식별자(예: ORCID)가 있다면 추가하는 것이 좋음
    full_name = StringProperty(index=True, required=True)
    last_name = StringProperty(index=True)
    fore_name = StringProperty()
    initials = StringProperty()
    # 소속 정보는 복잡하므로 단순 문자열 또는 별도 노드로 모델링 가능
    affiliation = StringProperty()
    articles = RelationshipTo('Article', 'AUTHORED_BY') # Author -> Article 관계

class Article(StructuredNode):
    uid = UniqueIdProperty() # neomodel 자동 생성 고유 ID
    pmid = StringProperty(unique_index=True, required=True)
    doi = StringProperty(index=True)
    title = StringProperty(required=True, max_length=500) # 최대 길이 제한 추가
    abstract = StringProperty()
    publication_year = IntegerProperty(index=True)
    publication_date = DateProperty() # 전체 날짜 (YYYY-MM-DD 형식)
    
    # 임베딩 벡터를 저장할 속성
    title_embedding = JSONProperty()  # 제목 임베딩
    abstract_embedding = JSONProperty()  # 초록 임베딩
    combined_embedding = JSONProperty()  # 제목+초록 통합 임베딩

    # 관계 정의
    authors = RelationshipFrom(Author, 'AUTHORED_BY') # Article <- Author 관계
    journal = RelationshipTo(Journal, 'PUBLISHED_IN')
    keywords = RelationshipTo(Keyword, 'HAS_KEYWORD')
    # cites = Relationship('Article', 'CITES') # 인용 관계 (선택적)

    def __str__(self):
        return f"{self.title} (PMID: {self.pmid})"

# 스키마 인덱스 및 제약 조건 설치 함수 (스크립트 또는 manage.py 명령어 사용 권장)
def install_neomodel_labels():
    """neomodel 모델 정의에 따른 인덱스 및 제약 조건을 Neo4j에 설치합니다."""
    print("Neomodel 레이블 및 제약 조건 설치 중...")
    from neomodel import install_labels
    try:
        install_labels(Article)
        install_labels(Author)
        install_labels(Journal)
        install_labels(Keyword)
        # install_labels(Authorship) # 관계 모델은 직접 설치 X
        print("Neomodel 레이블 및 제약 조건 설치 완료.")
    except Exception as e:
        print(f"레이블 설치 중 오류 발생: {e}")
        print("Neo4j 서버가 실행 중이고 연결 정보가 올바른지 확인하세요.")

# 이 스크립트가 직접 실행될 때 레이블 설치 (예시)
# if __name__ == "__main__":
#     install_neomodel_labels()
# Django 프로젝트에서는 manage.py install_labels 사용 권장 (django-neomodel 필요)
# 또는 별도 스크립트에서 호출