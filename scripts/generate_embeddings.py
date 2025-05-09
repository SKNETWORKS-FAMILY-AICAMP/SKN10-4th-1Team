#!/usr/bin/env python
# scripts/generate_embeddings.py

import os
import sys
from dotenv import load_dotenv
import time
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# .env 파일 로드
load_dotenv()

# Neo4j 모델 가져오기
from api.models import Article, db

# OpenAI API 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("오류: .env 파일에 OPENAI_API_KEY 환경 변수를 설정하세요.")
    sys.exit(1)

# OpenAI API 설정
from openai import OpenAI

def create_embedding(text, model="text-embedding-3-large", retry_count=3):
    """
    텍스트 데이터를 OpenAI API를 사용하여 임베딩합니다.
    
    Args:
        text (str): 임베딩할 텍스트
        model (str): 사용할 임베딩 모델
        retry_count (int): 재시도 횟수
    
    Returns:
        list: 임베딩 벡터, 에러 발생 시 None
    """
    if not text:
        return None
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    for attempt in range(retry_count):
        try:
            response = client.embeddings.create(
                input=text,
                model=model
            )
            
            # 임베딩 벡터 반환
            return response.data[0].embedding
        
        except Exception as e:
            # 마지막 시도에서도 실패하면 오류 출력 후 None 반환
            if attempt == retry_count - 1:
                print(f"임베딩 오류: {e}")
                return None
            
            # 재시도 전 약간의 대기 시간
            wait_time = (attempt + 1) * 2  # 점진적 백오프
            print(f"임베딩 오류: {e}. {wait_time}초 후 재시도 ({attempt+1}/{retry_count})...")
            time.sleep(wait_time)

def generate_embeddings_for_all_articles(batch_size=10):
    """
    Neo4j 데이터베이스에 저장된 모든 Article 노드에 대해 임베딩을 생성합니다.
    
    Args:
        batch_size (int): 한 번에 처리할 노드 수
    """
    print("Article 노드에 대한 임베딩 생성을 시작합니다...")
    
    # 총 Article 수 확인
    query = "MATCH (a:Article) RETURN count(a) as count"
    result, _ = db.cypher_query(query)
    total_count = result[0][0]
    print(f"총 {total_count}개의 Article 노드 임베딩 처리 예정")
    
    # 임베딩이 없는 Article 노드 가져오기
    query = """
    MATCH (a:Article) 
    WHERE a.combined_embedding IS NULL
    RETURN a.uid as uid, a.pmid as pmid, a.title as title, a.abstract as abstract
    LIMIT $batch_size
    """
    
    processed_count = 0
    success_count = 0
    error_count = 0
    
    while processed_count < total_count:
        # 배치로 처리할 노드 가져오기
        result, _ = db.cypher_query(query, {"batch_size": batch_size})
        
        if not result:
            print("더 이상 처리할 Article 노드가 없습니다.")
            break
        
        current_batch_size = len(result)
        print(f"\n배치 처리 중: {processed_count + 1} - {processed_count + current_batch_size} / {total_count}")
        
        for row in tqdm(result, desc="임베딩 생성"):
            uid, pmid, title, abstract = row
            
            # 제목과 초록이 모두 비어있다면 건너뜀
            if not title and not abstract:
                print(f"경고: PMID {pmid}의 제목과 초록이 모두 비어 있습니다. 건너뜁니다.")
                processed_count += 1
                error_count += 1
                continue
            
            # 제목과 초록이 있는 경우
            try:
                # 제목 임베딩
                title_embedding = create_embedding(title) if title else None
                
                # 초록 임베딩
                abstract_embedding = create_embedding(abstract) if abstract else None
                
                # 제목 + 초록 통합 임베딩
                combined_text = ""
                if title:
                    combined_text += title
                if abstract:
                    if combined_text:
                        combined_text += " "
                    combined_text += abstract
                
                combined_embedding = create_embedding(combined_text) if combined_text else None
                
                # 임베딩 저장
                update_query = """
                MATCH (a:Article {uid: $uid})
                SET a.title_embedding = $title_embedding,
                    a.abstract_embedding = $abstract_embedding,
                    a.combined_embedding = $combined_embedding
                """
                
                db.cypher_query(
                    update_query, 
                    {
                        "uid": uid,
                        "title_embedding": title_embedding,
                        "abstract_embedding": abstract_embedding,
                        "combined_embedding": combined_embedding
                    }
                )
                
                success_count += 1
                
                # API 호출 제한 방지를 위한 대기
                time.sleep(0.5)
            
            except Exception as e:
                print(f"오류: PMID {pmid} 처리 중 오류 발생: {e}")
                error_count += 1
            
            processed_count += 1
    
    print("\n임베딩 생성 완료:")
    print(f"성공: {success_count}개 Article")
    print(f"오류/건너뜀: {error_count}개 Article")

def add_embedding_to_ingest_data():
    """
    ingest_data.py 스크립트를 수정하여 데이터 적재 시 임베딩 생성 코드를 추가합니다.
    """
    # 이 함수는 ingest_data.py 스크립트에 임베딩 관련 코드를 추가하는 로직을 구현할 수 있습니다
    # 현재 구현에서는 생략하고 직접 스크립트 내용을 추가합니다.
    pass

if __name__ == "__main__":
    print("OpenAI 임베딩 생성 도구")
    print("=====================")
    print("1. 이미 저장된 모든 Article 노드에 대해 임베딩 생성")
    print("2. 특정 PMID의 Article에 대해 임베딩 생성")
    print("3. 종료")
    
    choice = input("\n선택하세요 (1-3): ")
    
    if choice == '1':
        batch_size = input("배치 크기를 입력하세요 (기본값: 10): ")
        batch_size = int(batch_size) if batch_size.isdigit() else 10
        generate_embeddings_for_all_articles(batch_size)
    
    elif choice == '2':
        pmid = input("PMID를 입력하세요: ")
        article = Article.nodes.filter(pmid=pmid).first()
        
        if not article:
            print(f"PMID {pmid}에 해당하는 Article을 찾을 수 없습니다.")
            sys.exit(1)
        
        title = article.title
        abstract = article.abstract
        
        print(f"Article 정보:")
        print(f"PMID: {article.pmid}")
        print(f"제목: {title}")
        print(f"초록 길이: {len(abstract) if abstract else 0}")
        
        # 임베딩 생성
        title_embedding = create_embedding(title) if title else None
        abstract_embedding = create_embedding(abstract) if abstract else None
        
        # 제목 + 초록 통합 임베딩
        combined_text = ""
        if title:
            combined_text += title
        if abstract:
            if combined_text:
                combined_text += " "
            combined_text += abstract
        
        combined_embedding = create_embedding(combined_text) if combined_text else None
        
        # 임베딩 저장
        article.title_embedding = title_embedding
        article.abstract_embedding = abstract_embedding
        article.combined_embedding = combined_embedding
        article.save()
        
        print("임베딩 생성 및 저장 완료!")
        print(f"제목 임베딩 크기: {len(title_embedding) if title_embedding else 0}")
        print(f"초록 임베딩 크기: {len(abstract_embedding) if abstract_embedding else 0}")
        print(f"통합 임베딩 크기: {len(combined_embedding) if combined_embedding else 0}")
    
    else:
        print("프로그램을 종료합니다.")
        sys.exit(0) 