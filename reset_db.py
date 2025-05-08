#### 그래프 db 초기화용 실행시 다같이 죽음


#!/usr/bin/env python
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 환경 변수 설정
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 로드
os.environ['NEO4J_BOLT_URL'] = os.getenv('NEO4J_BOLT_URL')
os.environ['USE_EMBEDDINGS'] = os.getenv('USE_EMBEDDINGS')

# 기존 db 설정 업데이트
from neomodel import config
config.DATABASE_URL = os.environ['NEO4J_BOLT_URL']

print(f"데이터베이스 연결 설정이 업데이트되었습니다: {config.DATABASE_URL}")

try:
    from neomodel import db
    # 데이터베이스 연결 테스트
    result, _ = db.cypher_query("MATCH (n) RETURN count(n) as count")
    node_count = result[0][0]
    print(f"현재 데이터베이스에 {node_count}개의 노드가 있습니다.")
    
    # 모든 노드 삭제 여부 확인
    confirm = input("모든 데이터를 삭제하시겠습니까? (y/n): ")
    if confirm.lower() == 'y':
        # 모든 제약 조건 삭제
        print("모든 제약 조건 삭제 중...")
        constraints_query = "SHOW CONSTRAINTS"
        try:
            constraints, _ = db.cypher_query(constraints_query)
            for constraint in constraints:
                constraint_name = constraint[0]
                drop_constraint_query = f"DROP CONSTRAINT {constraint_name}"
                db.cypher_query(drop_constraint_query)
                print(f"제약 조건 삭제됨: {constraint_name}")
        except Exception as e:
            print(f"제약 조건 삭제 중 오류 발생: {e}")
            print("계속 진행합니다...")
        
        # 모든 인덱스 삭제
        print("모든 인덱스 삭제 중...")
        indexes_query = "SHOW INDEXES"
        try:
            indexes, _ = db.cypher_query(indexes_query)
            for index in indexes:
                index_name = index[0]
                drop_index_query = f"DROP INDEX {index_name}"
                db.cypher_query(drop_index_query)
                print(f"인덱스 삭제됨: {index_name}")
        except Exception as e:
            print(f"인덱스 삭제 중 오류 발생: {e}")
            print("계속 진행합니다...")
        
        # 모든 노드와 관계 삭제
        print("모든 노드와 관계 삭제 중...")
        db.cypher_query("MATCH (n) DETACH DELETE n")
        print("모든 노드와 관계가 삭제되었습니다.")
        
        # 라벨 정보 확인
        print("라벨 정보 확인 중...")
        labels_query = "CALL db.labels()"
        labels, _ = db.cypher_query(labels_query)
        if labels:
            print("남아있는 라벨:")
            for label in labels:
                print(f"- {label[0]}")
            
            # 라벨 제거 시도
            print("\n라벨 제거 시도 중...")
            
            # 방법 1: 스키마 초기화 시도
            try:
                print("스키마 초기화 시도...")
                db.cypher_query("CALL db.schema.visualization()")
                db.cypher_query("CALL db.schema.nodeTypeProperties()")
                
                # 라벨별로 남은 노드가 있는지 확인하고 삭제
                for label in labels:
                    label_name = label[0]
                    db.cypher_query(f"MATCH (n:{label_name}) DELETE n")
                
                print("스키마 초기화 완료!")
            except Exception as e:
                print(f"스키마 초기화 오류: {e}")
            
            # 방법 2: APOC 라이브러리 사용 시도 (Neo4j에 APOC가 설치되어 있어야 함)
            try:
                print("APOC 라이브러리를 사용한 메타데이터 정리 시도...")
                db.cypher_query("CALL apoc.meta.stats()")
                db.cypher_query("CALL apoc.meta.graphSample()")
                # 실제 라벨 제거 작업을 위한 APOC 프로시저 (Neo4j 버전에 따라 지원되지 않을 수 있음)
                db.cypher_query("CALL apoc.schema.assert({}, {})")
                print("APOC 메타데이터 정리 완료!")
            except Exception as e:
                print(f"APOC 라이브러리 사용 오류: {e}")
                print("APOC 라이브러리가 설치되어 있지 않거나 사용할 수 없습니다.")
            
            # 방법 3: 개별 라벨에 대한 제거 시도
            try:
                print("개별 라벨 제거 시도...")
                for label in labels:
                    label_name = label[0]
                    # 라벨이 있는 모든 노드를 찾아 해당 라벨만 제거 (노드는 유지)
                    remove_label_query = f"""
                    MATCH (n:{label_name})
                    REMOVE n:{label_name}
                    """
                    db.cypher_query(remove_label_query)
                    print(f"라벨 제거 시도: {label_name}")
                print("개별 라벨 제거 완료!")
            except Exception as e:
                print(f"개별 라벨 제거 오류: {e}")
            
            # 라벨이 실제로 제거되었는지 다시 확인
            print("\n최종 라벨 확인 중...")
            try:
                labels, _ = db.cypher_query("CALL db.labels()")
                if labels:
                    print("여전히 남아있는 라벨:")
                    for label in labels:
                        print(f"- {label[0]}")
                    print("\n라벨을 완전히 제거하려면 Neo4j 데이터베이스를 재시작하거나,")
                    print("Neo4j 관리자 도구를 사용하여 데이터베이스를 초기화해야 할 수 있습니다.")
                else:
                    print("모든 라벨이 성공적으로 제거되었습니다!")
            except Exception as e:
                print(f"라벨 확인 오류: {e}")
        else:
            print("라벨이 모두 초기화되었습니다.")
        
        print("\n이제 ingest_data.py를 실행하여 임베딩과 함께 데이터를 다시 적재할 수 있습니다.")
    else:
        print("데이터 삭제가 취소되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}") 