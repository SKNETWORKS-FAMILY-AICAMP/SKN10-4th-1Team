import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from openai import OpenAI
from neomodel import db
import numpy as np

# 환경 변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

class Neo4jVectorSearch:
    """Neo4j를 사용한 벡터 검색 클래스"""
    
    def __init__(self, embedding_model="text-embedding-3-large"):
        """
        Neo4j 벡터 검색 초기화
        
        Args:
            embedding_model: 사용할 OpenAI 임베딩 모델
        """
        self.embedding_model = embedding_model
    
    def create_embedding(self, text: str) -> List[float]:
        """
        OpenAI API를 사용하여 텍스트의 임베딩을 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 벡터
        """
        if not text:
            return []
        
        try:
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        
        Args:
            vec1: 첫 번째 벡터
            vec2: 두 번째 벡터
            
        Returns:
            코사인 유사도 (0~1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # 문자열로 저장된 벡터를 숫자 배열로 변환
        if isinstance(vec1, str):
            try:
                # 문자열 형식 "[0.1, 0.2, ...]"에서 실수 리스트로 변환
                vec1 = vec1.strip('[]').split(',')
                vec1 = [float(x.strip()) for x in vec1]
            except:
                print(f"벡터 변환 오류 (vec1): {vec1[:100]}...")
                return 0.0
                
        if isinstance(vec2, str):
            try:
                # 문자열 형식 "[0.1, 0.2, ...]"에서 실수 리스트로 변환
                vec2 = vec2.strip('[]').split(',')
                vec2 = [float(x.strip()) for x in vec2]
            except:
                print(f"벡터 변환 오류 (vec2): {vec2[:100]}...")
                return 0.0
        
        vec1 = np.array(vec1, dtype=np.float64)
        vec2 = np.array(vec2, dtype=np.float64)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        의미적 검색 실행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최상위 결과 수
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 임베딩 생성
        query_embedding = self.create_embedding(query)
        if not query_embedding:
            return []
        
        # Neo4j에서 모든 Article 노드와 임베딩 가져오기
        cypher_query = """
        MATCH (a:Article)
        WHERE a.combined_embedding IS NOT NULL
        RETURN a.uid as uid, a.pmid as pmid, a.title as title, 
               a.abstract as abstract, a.combined_embedding as embedding
        """
        
        try:
            results, _ = db.cypher_query(cypher_query)
            
            if not results:
                return []
            
            # 유사도 계산 및 상위 K개 결과 반환
            similarities = []
            for row in results:
                uid, pmid, title, abstract, embedding = row
                
                if embedding:
                    similarity = self.cosine_similarity(query_embedding, embedding)
                    similarities.append({
                        'uid': uid,
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract,
                        'similarity': similarity
                    })
            
            # 유사도 내림차순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 상위 K개 반환
            return similarities[:top_k]
            
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
            return []
    
    def get_related_nodes(self, pmid: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        특정 논문과 관련된 노드들(저자, 키워드, 저널 등) 가져오기
        
        Args:
            pmid: 논문 PMID
            
        Returns:
            관련 노드 정보
        """
        related_info = {
            'authors': [],
            'keywords': [],
            'journal': None
        }
        
        # 저자 가져오기
        authors_query = """
        MATCH (author:Author)-[:AUTHORED_BY]->(a:Article {pmid: $pmid})
        RETURN author.full_name as full_name, author.affiliation as affiliation
        """
        
        # 키워드 가져오기
        keywords_query = """
        MATCH (a:Article {pmid: $pmid})-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.term <> 'COVID-19'
        RETURN k.term as term
        """
        
        # 저널 가져오기
        journal_query = """
        MATCH (a:Article {pmid: $pmid})-[:PUBLISHED_IN]->(j:Journal)
        RETURN j.name as name, j.issn as issn
        """
        
        try:
            # 저자 정보 가져오기
            author_results, _ = db.cypher_query(authors_query, {'pmid': pmid})
            for row in author_results:
                related_info['authors'].append({
                    'full_name': row[0],
                    'affiliation': row[1]
                })
            
            # 키워드 정보 가져오기
            keyword_results, _ = db.cypher_query(keywords_query, {'pmid': pmid})
            for row in keyword_results:
                related_info['keywords'].append({
                    'term': row[0]
                })
            
            # 저널 정보 가져오기
            journal_results, _ = db.cypher_query(journal_query, {'pmid': pmid})
            if journal_results and journal_results[0]:
                related_info['journal'] = {
                    'name': journal_results[0][0],
                    'issn': journal_results[0][1]
                }
            
            return related_info
        
        except Exception as e:
            print(f"관련 노드 가져오기 오류: {e}")
            return related_info
    
    def find_article_connections(self, pmid: str, degrees: int = 2) -> List[Dict[str, Any]]:
        """
        특정 논문과의 연결 관계 (2-3도 연결) 탐색
        
        Args:
            pmid: 논문 PMID
            degrees: 탐색할 연결 단계 수
            
        Returns:
            연결된 논문 정보
        """
        connection_query = f"""
    // 키워드 공유 관계
    MATCH (source:Article {{pmid: $pmid}})-[r1:HAS_KEYWORD]->(shared:Keyword)<-[r2:HAS_KEYWORD]-(target:Article)
    WHERE target.pmid <> $pmid AND shared.term <> 'COVID-19'
    RETURN target.pmid as target_pmid, 
           target.title as target_title, 
           target.abstract as target_abstract,
           labels(shared)[0] as shared_type,
           shared.term as shared_name,
           TYPE(r1) as source_relation,
           TYPE(r2) as target_relation
    LIMIT 10
    
    UNION
    
    // 저널 공유 관계
    MATCH (source:Article {{pmid: $pmid}})-[r1:PUBLISHED_IN]->(shared:Journal)<-[r2:PUBLISHED_IN]-(target:Article)
    WHERE target.pmid <> $pmid
    RETURN target.pmid as target_pmid, 
           target.title as target_title, 
           target.abstract as target_abstract,
           labels(shared)[0] as shared_type,
           shared.name as shared_name,
           TYPE(r1) as source_relation,
           TYPE(r2) as target_relation
    LIMIT 10
    
    UNION
    
    // 저자 공유 관계 (Author->Article 형태)
    MATCH (shared:Author)-[r1:AUTHORED_BY]->(source:Article {{pmid: $pmid}})
    MATCH (shared)-[r2:AUTHORED_BY]->(target:Article)
    WHERE target.pmid <> $pmid
    RETURN target.pmid as target_pmid, 
           target.title as target_title, 
           target.abstract as target_abstract,
           labels(shared)[0] as shared_type,
           shared.full_name as shared_name,
           TYPE(r1) as source_relation,
           TYPE(r2) as target_relation
    LIMIT 10
        """
        
        try:
            results, _ = db.cypher_query(connection_query, {'pmid': pmid})
            
            connections = []
            for row in results:
                target_pmid, target_title, target_abstract, shared_type, shared_name, source_relation, target_relation = row
                
                # 연결 정보 구성
                connection_info = {
                    'pmid': target_pmid,
                    'title': target_title,
                    'abstract': target_abstract,
                    'shared_node': {
                        'type': shared_type, 
                        'name': shared_name
                    },
                    'source_relation': source_relation,
                    'target_relation': target_relation,
                    'connection_description': f"{source_relation} -> {shared_type}:{shared_name} -> {target_relation}"
                }
                
                connections.append(connection_info)
            
            return connections
        
        except Exception as e:
            print(f"연결 관계 탐색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def graph_based_reranking(self, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        그래프 관계를 활용한 검색 결과 리랭킹
        
        Args:
            docs: 벡터 검색으로 찾은 문서 리스트
            top_k: 반환할 최종 상위 결과 수
            
        Returns:
            리랭킹된 결과 리스트
        """
        if not docs or len(docs) <= 1:
            return docs[:top_k]
        
        # 모든 문서의 PMID 추출
        pmids = [doc.get('pmid') for doc in docs if doc.get('pmid')]
        if not pmids:
            return docs[:top_k]
        
        # 그래프 연결성 분석을 위한 쿼리
        graph_query = """
        MATCH (a:Article)
        WHERE a.pmid IN $pmids
        
        // 저자 연결 수
        OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(auth:Author)
        WITH a, count(auth) as author_count
        
        // 키워드 연결 수
        OPTIONAL MATCH (a)-[:HAS_KEYWORD]->(k:Keyword)
        WITH a, author_count, count(k) as keyword_count
        
        // 인용 관계 (현재는 프로젝트에서 사용하지 않음)
        WITH a, author_count, keyword_count, 0 as citations_out, 0 as citations_in
        
        // 공통 저자 수
        OPTIONAL MATCH (a)<-[:AUTHORED_BY]-(auth:Author)-[:AUTHORED_BY]->(other:Article)
        WHERE other.pmid IN $pmids AND other.pmid <> a.pmid
        WITH a, author_count, keyword_count, citations_out, citations_in, 
             count(DISTINCT other) as shared_authors_count
        
        // 공통 키워드 수
        OPTIONAL MATCH (a)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(other:Article)
        WHERE other.pmid IN $pmids AND other.pmid <> a.pmid
        
        RETURN a.pmid as pmid, 
               author_count, 
               keyword_count,
               citations_out,
               citations_in,
               shared_authors_count,
               count(DISTINCT other) as shared_keywords_count
        """
        
        try:
            # Neo4j 쿼리 실행
            results, _ = db.cypher_query(graph_query, {'pmids': pmids})
            
            if not results:
                return docs[:top_k]
            
            # 그래프 점수 계산 및 매핑
            graph_scores = {}
            for row in results:
                pmid, author_conn, keyword_conn, cite_out, cite_in, shared_auth, shared_kw = row
                
                # 가중치 설정
                author_weight = 1.0
                keyword_weight = 0.8
                citation_in_weight = 1.5  # 인용 받은 것(피인용)이 더 중요
                citation_out_weight = 0.7
                shared_authors_weight = 1.2
                shared_keywords_weight = 0.9
                
                # 그래프 연결성 점수 계산
                graph_score = (
                    author_conn * author_weight + 
                    keyword_conn * keyword_weight +
                    cite_out * citation_out_weight + 
                    cite_in * citation_in_weight +
                    shared_auth * shared_authors_weight + 
                    shared_kw * shared_keywords_weight
                )
                
                graph_scores[pmid] = graph_score
            
            # 최대 그래프 점수로 정규화
            max_graph_score = max(graph_scores.values()) if graph_scores else 1.0
            for pmid in graph_scores:
                graph_scores[pmid] /= max_graph_score
            
            # 벡터 유사도와 그래프 점수 결합
            reranked_docs = []
            for doc in docs:
                pmid = doc.get('pmid')
                vector_similarity = doc.get('similarity', 0)
                
                # 그래프 점수가 없으면 기본값 사용
                graph_score = graph_scores.get(pmid, 0)
                
                # 결합 가중치 설정 (벡터 유사도 70%, 그래프 점수 30%)
                vector_weight = 0.7
                graph_weight = 0.3
                
                # 최종 점수 계산
                final_score = (vector_similarity * vector_weight) + (graph_score * graph_weight)
                
                # 새 문서 정보 구성
                reranked_doc = doc.copy()
                reranked_doc['original_similarity'] = vector_similarity
                reranked_doc['graph_score'] = graph_score
                reranked_doc['similarity'] = final_score  # 최종 점수로 유사도 업데이트
                
                reranked_docs.append(reranked_doc)
            
            # 최종 점수로 정렬
            reranked_docs.sort(key=lambda x: x['similarity'], reverse=True)
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            print(f"그래프 기반 리랭킹 오류: {e}")
            return docs[:top_k]
    
    def semantic_search_with_reranking(self, query: str, initial_k: int = 40, final_k: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행 후 그래프 및 LLM 기반 리랭킹 적용
        
        Args:
            query: 검색 쿼리
            initial_k: 초기 벡터 검색 결과 수
            final_k: 최종 반환할 결과 수
            
        Returns:
            리랭킹된 검색 결과 리스트
        """
        # 두 단계 검색 파이프라인 실행
        return self.two_stage_retrieval(query, initial_k, final_k)

    def two_stage_retrieval(self, query: str, initial_k: int = 40, final_k: int = 5) -> List[Dict[str, Any]]:
        """
        2단계 검색 파이프라인: 벡터 검색 후 LLM 리랭킹
        그래프 연결 관계는 최종 응답 생성 단계에서만 활용
        
        Args:
            query: 한국어 검색 쿼리
            initial_k: 초기 벡터 검색 결과 수
            final_k: 최종 반환할 결과 수
            
        Returns:
            LLM 리랭킹된 검색 결과 리스트
        """
        # 1단계: 쿼리 번역 및 벡터 검색
        try:
            # 한국어 쿼리를 영어로 번역
            translated_query = self._translate_query_with_llm(query)
            print(f"원본 쿼리: {query}")
            print(f"번역된 쿼리: {translated_query}")
            
            # 번역된 쿼리로 벡터 검색 수행
            initial_results = self.semantic_search(translated_query, top_k=initial_k)
            
            if not initial_results:
                return []
            
            # 2단계: LLM 기반 리랭킹
            reranked_results = self._llm_only_reranking(query, initial_results, top_k=final_k)
            
            return reranked_results
            
        except Exception as e:
            print(f"2단계 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _translate_query_with_llm(self, query: str) -> str:
        """
        LLM을 사용하여 한국어 쿼리를 영어로 번역
        """
        try:
            import google.generativeai as genai
            
            # Gemini API 설정
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("GEMINI_API_KEY가 설정되지 않아 쿼리 번역을 건너뜁니다.")
                return query
                
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # 번역 프롬프트
            prompt = f"""
한국어를 영어로 정확하게 번역해주세요. 의미가 그대로 유지되도록 번역하고, 의학/과학적 전문용어가 있다면 정확한 영어 용어로 번역해주세요.
번역할 텍스트: {query}

영어 번역:
"""
            
            response = model.generate_content(prompt)
            translated_text = response.text.strip()
            
            # 불필요한 접두사 제거
            prefixes_to_remove = ["영어 번역:", "Translation:", "English translation:"]
            for prefix in prefixes_to_remove:
                if translated_text.startswith(prefix):
                    translated_text = translated_text[len(prefix):].strip()
            
            return translated_text
        
        except Exception as e:
            print(f"쿼리 번역 오류: {e}")
            return query  # 오류 시 원본 쿼리 반환

    def _llm_only_reranking(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        LLM만 사용하여 검색 결과 리랭킹 (그래프 관계 반영하지 않음)
        
        Args:
            query: 한국어 검색 쿼리
            docs: 벡터 검색으로 찾은 문서 리스트
            top_k: 반환할 최종 상위 결과 수
            
        Returns:
            리랭킹된 결과 리스트
        """
        if not docs or len(docs) <= 1:
            return docs[:top_k]
        
        try:
            import google.generativeai as genai
            
            # 응답 스키마 정의
            class DocumentScore(TypedDict):
                score: int  # 0-10 사이의 점수
            
            # Gemini API 설정
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                print("GEMINI_API_KEY가 설정되지 않아 LLM 리랭킹을 건너뜁니다.")
                return docs[:top_k]
                
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # 평가할 문서 선택 (최대 10개)
            evaluation_docs = docs[:10] if len(docs) > 10 else docs
            
            # LLM 평가 점수 저장
            llm_scores = {}
            
            # 평가 프롬프트 템플릿
            prompt_template = """
당신은 한국어 쿼리에 대해 영어 학술 문서의 관련성을 평가하는 전문가입니다.
다음 한국어 질문과 영어 문서의 관련성을 0에서 10 사이의 점수로 평가해주세요.
10은 매우 관련성이 높음을, 0은 전혀 관련 없음을 의미합니다.

## 한국어 질문:
{query}

## 영어 문서:
제목: {title}
요약: {abstract}

이 영어 문서가 한국어 질문과 얼마나 관련이 있는지 평가하세요.
문서의 내용이 한국어 질문의 의도와 관련이 있는지 주의 깊게 판단하세요.
score는 반드시 0부터 10 사이의 정수만 입력하세요.
"""
            
            # 각 문서 평가
            for doc in evaluation_docs:
                pmid = doc.get('pmid')
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                
                if not abstract:
                    abstract = "내용 없음"
                elif len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                
                # 프롬프트 생성
                prompt = prompt_template.format(
                    query=query,
                    title=title,
                    abstract=abstract
                )
                
                # LLM 호출 및 점수 획득
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.1,
                            "response_mime_type": "application/json",
                            "response_schema": DocumentScore
                        }
                    )
                    
                    if hasattr(response, 'candidates') and response.candidates:
                        content = response.candidates[0].content
                        if hasattr(content, 'parts') and content.parts:
                            result = content.parts[0].text
                            import json
                            try:
                                result_dict = json.loads(result)
                                if 'score' in result_dict:
                                    score = int(result_dict['score'])
                                    if 0 <= score <= 10:
                                        llm_scores[pmid] = score / 10.0  # 0~1 범위로 정규화
                                        print(f"문서 평가 (PMID: {pmid}): 점수 {score}/10")
                                    else:
                                        llm_scores[pmid] = 0.5
                                else:
                                    llm_scores[pmid] = 0.5
                            except json.JSONDecodeError:
                                llm_scores[pmid] = 0.5
                        else:
                            llm_scores[pmid] = 0.5
                    else:
                        llm_scores[pmid] = 0.5
                
                except Exception as e:
                    print(f"LLM 평가 오류 (PMID: {pmid}): {e}")
                    llm_scores[pmid] = 0.5
            
            # 리랭킹 수행
            reranked_docs = []
            for doc in docs:
                pmid = doc.get('pmid')
                original_similarity = doc.get('similarity', 0)
                
                # LLM 점수 (평가되지 않은 문서는 원래 유사도 사용)
                if pmid in llm_scores:
                    new_score = llm_scores[pmid]
                else:
                    new_score = original_similarity
                
                # 새 문서 정보 구성
                reranked_doc = doc.copy()
                reranked_doc['original_similarity'] = original_similarity
                reranked_doc['llm_score'] = new_score
                reranked_doc['similarity'] = new_score  # LLM 점수로 유사도 업데이트
                
                reranked_docs.append(reranked_doc)
            
            # 최종 점수로 정렬
            reranked_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return reranked_docs[:top_k]
            
        except Exception as e:
            print(f"LLM 리랭킹 오류: {e}")
            import traceback
            traceback.print_exc()
            return docs[:top_k]

    def get_graph_context_for_response(self, selected_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LLM 응답 생성을 위한 그래프 연결 관계 정보 추출
        
        Args:
            selected_docs: 선택된 상위 문서 리스트
            
        Returns:
            응답 생성에 활용할 그래프 컨텍스트 정보
        """
        context = {
            'document_connections': [],
            'authors': {},
            'keywords': {},
            'topics': {}
        }
        
        if not selected_docs:
            return context
        
        try:
            # 선택된 문서의 PMID 수집
            pmids = [doc.get('pmid') for doc in selected_docs if doc.get('pmid')]
            if not pmids:
                return context
            
            # 1. 문서 간 연결 관계 탐색
            connection_query = """
            MATCH (a:Article)
            WHERE a.pmid IN $pmids
            MATCH (b:Article)
            WHERE b.pmid IN $pmids AND a <> b
            
            OPTIONAL MATCH path = shortestPath((a)-[*1..3]-(b))
            WHERE length(path) > 0 AND ALL(n IN nodes(path) WHERE NOT (n:Keyword AND n.term = 'COVID-19'))
            WITH a, b, path, length(path) AS distance
            ORDER BY distance
            RETURN a.pmid AS source_pmid, b.pmid AS target_pmid,
                   [node in nodes(path) WHERE NOT node:Article | labels(node)[0] + ': ' + coalesce(node.name, node.term, node.full_name, 'Unknown')] AS connecting_nodes,
                   [rel in relationships(path) | type(rel)] AS relation_types,
                   distance
            LIMIT 20
            """
            
            results, _ = db.cypher_query(connection_query, {'pmids': pmids})
            
            for row in results:
                source_pmid, target_pmid, connecting_nodes, relation_types, distance = row
                context['document_connections'].append({
                    'source': source_pmid,
                    'target': target_pmid,
                    'connecting_entities': connecting_nodes,
                    'relations': relation_types,
                    'distance': distance
                })
            
            # 2. 주요 저자 정보
            authors_query = """
            MATCH (auth:Author)-[:AUTHORED_BY]->(a:Article)
            WHERE a.pmid IN $pmids
            WITH auth, count(a) AS doc_count
            ORDER BY doc_count DESC
            LIMIT 5
            RETURN auth.full_name AS name, auth.affiliation AS affiliation, doc_count
            """
            
            author_results, _ = db.cypher_query(authors_query, {'pmids': pmids})
            
            for row in author_results:
                name, affiliation, doc_count = row
                if name:
                    context['authors'][name] = {
                        'affiliation': affiliation,
                        'document_count': doc_count
                    }
            
            # 3. 주요 키워드 정보
            keywords_query = """
            MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE a.pmid IN $pmids AND k.term <> 'COVID-19'
            WITH k, count(a) AS usage_count
            ORDER BY usage_count DESC
            LIMIT 10
            RETURN k.term AS term, usage_count,
                   [] AS related_terms
            """
            
            keyword_results, _ = db.cypher_query(keywords_query, {'pmids': pmids})
            
            for row in keyword_results:
                term, usage_count, related_terms = row
                if term:
                    context['keywords'][term] = {
                        'usage_count': usage_count,
                        'related_terms': related_terms
                    }
            
            # 4. 관련 주제 정보 (주의: Topic 노드가 없으므로 키워드를 대체로 사용)
            topics_query = """
            MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE a.pmid IN $pmids AND k.term <> 'COVID-19'
            WITH k, count(a) AS doc_count
            ORDER BY doc_count DESC
            LIMIT 5
            RETURN k.term AS name, doc_count,
                   [] AS related_topics
            """
            
            topic_results, _ = db.cypher_query(topics_query, {'pmids': pmids})
            
            for row in topic_results:
                name, doc_count, related_topics = row
                if name:
                    context['topics'][name] = {
                        'document_count': doc_count,
                        'related_topics': related_topics
                    }
            
            return context
            
        except Exception as e:
            print(f"그래프 컨텍스트 추출 오류: {e}")
            import traceback
            traceback.print_exc()
            return context

# 단독 테스트용
if __name__ == "__main__":
    vector_search = Neo4jVectorSearch()
    results = vector_search.semantic_search("obesity treatment")
    print(f"검색 결과: {len(results)}개 찾음")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (유사도: {result['similarity']:.4f})") 