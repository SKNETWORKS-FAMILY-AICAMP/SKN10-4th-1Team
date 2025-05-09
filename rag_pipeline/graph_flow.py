import os
import time
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, TypedDict, Literal
import random
import json

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv



# 로컬 모듈 임포트
from rag_pipeline.vector_store import Neo4jVectorSearch
from rag_pipeline.llm import GeminiLLM
from rag_pipeline.tavily_search import TavilySearch
from rag_pipeline.neo4j_graph_chain import Neo4jGraphChain

# 환경 변수 로드
load_dotenv()

# 상태 클래스 정의
class QueryState(TypedDict):
    messages: List[Dict[str, str]]  # 대화 메시지 목록
    query: str  # 현재 쿼리
    query_type: str  # 쿼리 타입: "vector", "graph", "hybrid"
    vector_results: List[Dict]  # 벡터 검색 결과
    graph_results: List[Dict]  # 그래프 검색 결과
    combined_results: List[Dict]  # 결합된 결과
    related_nodes: Dict[str, Any]  # 관련 노드 정보
    graph_context: Dict[str, Any]  # 그래프 컨텍스트
    final_answer: str  # 최종 답변
    citations: List[Dict]  # 인용 정보
    graph_search_output: Dict[str, Any]  # 그래프 검색 출력 (샘플링된 쿼리 및 raw_results 포함)
    
    # 응답 검증 및 추가 검색 관련 필드
    response_quality: Literal["good", "insufficient", "unverified"]  # 응답 품질 평가
    tavily_results: List[Dict]  # Tavily 검색 결과
    additional_answer: str  # 추가 답변
    iterations: int  # 반복 횟수 (무한 루프 방지)


class HybridGraphFlow:
    """
    하이브리드 그래프 기반 RAG 검색 파이프라인
    
    LangGraph를 사용하여 그래프, 벡터, 하이브리드 검색 기능을 결합한 고급 RAG 시스템 구현
    """
    
    def __init__(self):
        """HybridGraphFlow 초기화"""
        self.vector_search = Neo4jVectorSearch()
        self.llm = GeminiLLM()
        try:
            self.graph_chain = Neo4jGraphChain() # Initialize Neo4jGraphChain
            if not self.graph_chain.cypher_chain: # Check if Neo4jGraphChain itself initialized correctly
                print("HybridGraphFlow __init__ Warning: self.graph_chain.cypher_chain is None. Neo4jGraphChain might not have initialized its QA chain properly.")
                # Depending on strictness, could raise an error or allow proceeding with graph_chain potentially non-functional
        except Exception as e:
            print(f"HybridGraphFlow __init__ Error: Failed to initialize Neo4jGraphChain: {e}")
            self.graph_chain = None # Set to None to indicate failure
        self.tavily_search = TavilySearch()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        
        # StateGraph 생성
        graph = StateGraph(QueryState)
        
        # 노드 추가
        graph.add_node("determine_query_type", self._determine_query_type)
        graph.add_node("vector_search", self._vector_search)
        graph.add_node("graph_search", self._graph_search)
        graph.add_node("hybrid_search", self._hybrid_search)
        graph.add_node("gather_related_nodes", self._gather_related_nodes)
        graph.add_node("extract_graph_context", self._extract_graph_context)
        graph.add_node("generate_response", self._generate_response)
        
        # 응답 검증 및 추가 검색 노드 추가
        graph.add_node("evaluate_response", self._evaluate_response)
        graph.add_node("tavily_search", self._tavily_search)
        graph.add_node("generate_additional_response", self._generate_additional_response)
        
        # 조건부 엣지 및 워크플로우 정의
        graph.add_conditional_edges(
            "determine_query_type",
            lambda state: state["query_type"],
            {
                "vector": "vector_search",
                "graph": "graph_search",
                "hybrid": "hybrid_search"
            }
        )
        
        # 순수 벡터 검색은 그래프 관련 노드 건너뜀기 (그래프 정보 참고하지 않음)
        graph.add_edge("vector_search", "generate_response")
        
        # 그래프/하이브리드 검색은 그래프 정보 추출 노드 포함
        graph.add_edge("graph_search", "gather_related_nodes")
        graph.add_edge("hybrid_search", "gather_related_nodes")
        graph.add_edge("gather_related_nodes", "extract_graph_context")
        graph.add_edge("extract_graph_context", "generate_response")
        
        # 응답 생성 후 검증 및 추가 검색 워크플로우 추가
        graph.add_edge("generate_response", "evaluate_response")
        
        # 응답 품질에 따른 조건부 엣지 추가
        graph.add_conditional_edges(
            "evaluate_response",
            lambda state: state["response_quality"],
            {
                "good": END,  # 좋은 응답이면 종료
                "insufficient": "tavily_search",  # 불충분한 응답이면 Tavily 검색
                "unverified": END  # 검증 불가능한 경우 (예: API 키 누락) 그냥 종료
            }
        )
        
        # Tavily 검색 후 추가 응답 생성
        graph.add_edge("tavily_search", "generate_additional_response")
        graph.add_edge("generate_additional_response", END)
        
        # 시작 노드 설정
        graph.set_entry_point("determine_query_type")
        
        # 컴파일 및 반환
        return graph.compile()
    
    def _determine_query_type(self, state: QueryState) -> QueryState:
        """
        사용자 쿼리 의도 분석하여 최적의 검색 전략 결정 (LLM 활용)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # LLM을 사용하여 쿼리 유형 결정
        from rag_pipeline.llm import GeminiLLM
        
        llm = GeminiLLM()
        
        prompt = f"""
        당신은 연구 논문 검색 시스템의, 쿼리 의도 분석을 담당하는 전문가입니다.
        사용자의 쿼리를 분석하여 적절한 검색 전략을 결정해야 합니다.

        다음 세 가지 검색 전략 중 하나를 선택하세요:
        1. "vector": 의미적 유사성에 기반한 검색이 필요할 때 (예: "당뇨병 치료에 관한 최신 연구", "COVID-19 백신의 효과")
        2. "graph": 논문/저자/키워드 간의 관계나 네트워크를 탐색할 때 (예: "Smith 교수와 공동 연구한 저자들", "면역학과 신경학 분야를 연결하는 연구")
        3. "hybrid": 복합적인 정보 요구가 있을 때 (기본값, 벡터 검색과 그래프 탐색을 모두 활용)

        사용자 쿼리: {query}

        위 쿼리에 가장 적합한 검색 전략은 무엇인가요? "vector", "graph", "hybrid" 중 하나만 출력하세요.
        """
        
        try:
            response = llm.model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # 유효한 응답인지 확인
            if result in ["vector", "graph", "hybrid"]:
                query_type = result
            else:
                # 유효하지 않은 응답인 경우 하이브리드 검색 사용
                query_type = "hybrid"
                print(f"LLM 응답이 유효하지 않음 ('{result}'), 기본값 'hybrid' 사용")
        except Exception as e:
            # LLM 호출 실패 시 하이브리드 검색 사용
            query_type = "hybrid"
            print(f"쿼리 유형 결정 중 오류 발생: {e}")
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _vector_search(self, state: QueryState) -> QueryState:
        """
        벡터 검색 수행 (그래프 정보 사용하지 않음)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # 벡터 검색 수행
        vector_results = self.vector_search.semantic_search(query, top_k=10)
        
        # 벡터 검색 모드에서는 각 문서에 검색 타입 표시
        for result in vector_results:
            result["search_type"] = "vector"
        
        # 그래프 관련 필드 초기화 - 벡터 검색에서는 그래프 정보 사용하지 않음
        return {
            **state,
            "vector_results": vector_results,
            "graph_results": [],  # 그래프 결과 빈 리스트로 초기화
            "combined_results": vector_results,  # 벡터 결과가 최종 결과
            "related_nodes": {},  # 관련 노드 정보 빈 사전으로 초기화
            "graph_context": {}   # 그래프 컨텍스트 빈 사전으로 초기화
        }
    
    def _graph_search(self, state: QueryState) -> QueryState:
        """
        그래프 검색 수행 - LangChain GraphCypherQAChain을 활용한 그래프 검색
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        print("\n" + "=" * 80)
        print(f"[디버깅] 사용자 쿼리: '{query}'")
        print("[디버깅] GraphCypherQAChain을 사용하여 그래프 검색 시작...")
        
        # LangChain GraphCypherQAChain 사용
        chain_result = self.graph_chain.search(query)
        
        # 생성된 Cypher 쿼리 출력
        cypher_query = chain_result.get("cypher_query", "")
        print(f"[디버깅] 생성된 Cypher 쿼리:\n{cypher_query}")

        # 검색 결과를 state에 저장
        state['graph_search_output'] = chain_result
        print(f"[디버깅] 그래프 검색 결과 (raw_results):\n{chain_result.get('raw_results')}")
        print(f"[디버깅] state에 저장된 graph_search_output: {state['graph_search_output']}")
        print("=" * 80)

        return state
    
    def _generate_cypher_query(self, user_query: str) -> str:
        """
        사용자 쿼리를 분석하여 적절한 Neo4j Cypher 쿼리 생성
        
        Args:
            user_query: 사용자 입력 쿼리
            
        Returns:
            실행할 Cypher 쿼리 문자열
        """
        if not self.graph_chain:
            print("Error in _generate_cypher_query: self.graph_chain was not successfully initialized.")
            return "Error: Graph chain service not available."

        # Further check if the internal QA chain of Neo4jGraphChain is ready
        if not self.graph_chain.cypher_chain:
             print("Error in _generate_cypher_query: Neo4jGraphChain's core component (cypher_chain) is not available.")
             return "Error: Graph query generation component not ready."

        try:
            # The search method of Neo4jGraphChain calls the QA chain and extracts the Cypher query.
            graph_search_output = self.graph_chain.search(user_query)
            
            generated_query = graph_search_output.get("cypher_query")

            # Check for various forms of failure indications in the generated query string
            if not generated_query or \
               "쿼리를 생성할 수 없습니다." in str(generated_query) or \
               (isinstance(generated_query, str) and "Error:" in generated_query):
                error_detail = str(generated_query) if generated_query else "No query returned"
                print(f"Warning: Cypher query generation failed or returned an error for user_query '{user_query}'. Detail: '{error_detail}'")
                return "Error: Cypher query generation resulted in an error or no query."
            
            return str(generated_query) # Ensure it's a string
        except Exception as e:
            print(f"Exception in _generate_cypher_query for query '{user_query}': {e}")
            # Return a generic error message to the caller, details are logged.
            return "Error: An exception occurred while generating the Cypher query."

    def _translate_query_if_needed(self, user_query: str) -> str:
        """
        한국어 쿼리를 영어로 번역하는 함수 - LLM 사용
        
        Args:
            user_query: 사용자 입력 쿼리
            
        Returns:
            영어로 번역된 쿼리
        """
        # 사용자 쿼리가 영어인지 한국어인지 간단히 확인
        # 영어 확떠이 높으면 그대로 유지
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,. ')
        ratio = sum(1 for c in user_query if c in english_chars) / len(user_query) if user_query else 0
        
        if ratio > 0.7:  # 영어 컨텐츠가 70% 이상이면 번역 없이 반환
            return user_query
        
        # LLM을 사용하여 번역
        try:
            llm = self.llm  # 기존 LLM 클래스 사용
            prompt = f"""
            Translate the following Korean medical query into English as accurately as possible,
            focusing on medical terminology and maintaining the original meaning.
            Only return the translated English text, nothing else.
            
            Korean query: {user_query}
            """
            
            response = llm.model.generate_content(prompt)
            translated = response.text.strip()
            
            print(f"[디버깅] 한국어 쿼리 번역: '{user_query}' -> '{translated}'")
            return translated
        except Exception as e:
            print(f"번역 오류: {e}, 원본 쿼리 사용")
            return user_query
    
    def _get_fallback_cypher_query(self, user_query: str) -> str:
        """
        LLM 쿼리 생성이 실패했을 때 사용할 대체 Cypher 쿼리
        
        Args:
            user_query: 사용자 입력 쿼리
            
        Returns:
            대체 Cypher 쿼리
        """
        # 번역 시도
        translated_query = self._translate_query_if_needed(user_query)
        
        # 쿼리에서 주요 키워드 추출 (간단한 구현)
        keywords = translated_query.lower().split()
        keywords = [k for k in keywords if len(k) > 3 and k not in 
                   ['what', 'which', 'when', 'where', 'how', 'why', 'who', 'and', 'the', 'for', 'with']]
        
        # 키워드가 없으면 일반 쿼리 반환
        if not keywords:
            return """
            MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
            WITH a, collect(k.term) as keywords
            RETURN a.pmid as pmid, a.title as title, a.abstract as abstract, keywords
            ORDER BY size(keywords) DESC, a.pmid DESC
            LIMIT 10
            """
        
        # 첫 번째 키워드로 검색 쿼리 구성
        keyword = keywords[0]
        return f"""
        MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE toLower(k.term) CONTAINS toLower('{keyword}')
        WITH a, collect(k.term) as keywords
        MATCH (a)-[:AUTHORED_BY]->(auth:Author)
        WITH a, keywords, collect(auth.name) as authors
        RETURN a.pmid as pmid, a.title as title, a.abstract as abstract, keywords, authors
        ORDER BY size(keywords) DESC
        LIMIT 15
        """
        
        try:
            # GraphCypherQAChain에서 반환한 결과 처리
            graph_results = chain_result.get("results", [])
            
            # 결과가 없으면 대체 질의 시도
            if not graph_results and cypher_query:
                print(f"[디버깅] GraphCypherQAChain 결과 없음, Cypher 쿼리 직접 실행...")
                start_time = time.time()
                results, _ = self.vector_search.db.cypher_query(cypher_query)
                query_time = time.time() - start_time
                print(f"[디버깅] 쿼리 실행 완료: {query_time:.2f}초 소요, {len(results)} 개의 결과 발견")
                
                # 결과 포맷팅 - 다양한 반환 구조 처리
                graph_results = []
                
                for row in results:
                    # 결과 구조에 따라 처리 방식 다르게 적용
                    if len(row) >= 4:  # 기본 형식 (pmid, title, abstract, keywords)
                        pmid, title, abstract, keywords = row[:4]
                        result_dict = {
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "keywords": keywords,
                            "search_type": "graph"
                        }
                        # 추가 데이터 있으면 처리 (저자 등)
                        if len(row) > 4:
                            result_dict["authors"] = row[4] if isinstance(row[4], list) else [row[4]]
                        if len(row) > 5:
                            result_dict["additional_data"] = row[5:]
                    else:
                        # 결과 구조가 예상과 다른 경우
                        print(f"예상치 못한 쿼리 결과 구조: {row}")
                        # 최소한의 정보만 포함
                        result_dict = {
                            "pmid": row[0] if len(row) > 0 else "unknown",
                            "title": row[1] if len(row) > 1 else "제목 없음",
                            "abstract": row[2] if len(row) > 2 else "초록 없음",
                            "keywords": row[3] if len(row) > 3 else [],
                            "search_type": "graph"  
                        }
                    
                    graph_results.append(result_dict)
            
            # 결과 메타데이터 추가 및 로깅
            print(f"[디버깅] 그래프 검색 결과: {len(graph_results)}건 발견")
            print("\n" + "=" * 40 + " 검색 결과 요약 " + "=" * 40)
            
            if graph_results:
                # 첫 번째 결과 제목 출력 (디버깅용)
                for i, result in enumerate(graph_results[:3]):  # 처음 3개 결과만 표시
                    title = result.get("title", "제목 없음")
                    pmid = result.get("pmid", "PMID 없음")
                    keywords = ", ".join(result.get("keywords", [])[:5])
                    print(f"\n결과 #{i+1} - PMID: {pmid}")
                    print(f"\u2022 제목: {title[:100]}{'...' if len(title) > 100 else ''}")
                    print(f"\u2022 키워드: {keywords}")
                
                print("\n" + "=" * 80)
            
            return {
                **state,
                "graph_results": graph_results,
                "combined_results": graph_results,  # 그래프 검색 모드에서는 그래프 결과가 최종 결과
                "graph_query_used": cypher_query  # 사용된 쿼리 기록 (디버깅용)
            }
        
        except Exception as e:
            print(f"그래프 검색 오류: {e}")
            return {
                **state,
                "graph_results": [],
                "combined_results": [],
                "graph_query_used": cypher_query,  # 오류가 발생한 쿼리도 기록
                "graph_error": str(e)  # 오류 메시지 저장
            }
    
    def _hybrid_search(self, state: QueryState) -> QueryState:
        """
        하이브리드 검색 수행 (벡터 + 그래프)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # 1. 먼저 벡터 검색 수행
        vector_results = self.vector_search.semantic_search(query, top_k=5)
        
        # 벡터 검색 결과가 없으면 종료
        if not vector_results:
            return {
                **state,
                "vector_results": [],
                "graph_results": [],
                "combined_results": []
            }
        
        # 2. 벡터 검색 결과를 기반으로 그래프 검색 수행
        pmids = [doc.get("pmid") for doc in vector_results if doc.get("pmid")]
        
        if not pmids:
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": [],
                "combined_results": vector_results
            }
        
        # 벡터 검색으로 찾은 문서들과 연결된 다른 문서 찾기
        try:
            connections = []
            for pmid in pmids[:3]:  # 상위 3개만 처리
                article_connections = self.vector_search.find_article_connections(pmid)
                connections.extend(article_connections)
            
            # 그래프 결과와 벡터 결과 결합
            combined_results = vector_results.copy()
            
            # 중복 제거하면서 그래프 결과 추가
            existing_pmids = set(pmids)
            for conn in connections:
                if conn.get("pmid") and conn.get("pmid") not in existing_pmids:
                    conn["search_type"] = "graph"  # 이 결과가 그래프 검색에서 나온 것임을 표시
                    combined_results.append(conn)
                    existing_pmids.add(conn.get("pmid"))
                    
                    # 최대 15개 결과로 제한
                    if len(combined_results) >= 15:
                        break
            
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": connections,
                "combined_results": combined_results
            }
        
        except Exception as e:
            print(f"하이브리드 검색 오류: {e}")
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": [],
                "combined_results": vector_results
            }
    
    def _gather_related_nodes(self, state: QueryState) -> QueryState:
        """
        검색 결과의 관련 노드 정보 수집
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        print(f"[디버깅] _gather_related_nodes 진입, state['graph_search_output']: {state.get('graph_search_output')}")
        combined_results = state.get("combined_results", [])
        
        if not combined_results:
            return {**state, "related_nodes": {}}
        
        # 첫 번째 문서의 PMID로 관련 노드 정보 가져오기
        first_doc_pmid = combined_results[0].get("pmid")
        if not first_doc_pmid:
            return {**state, "related_nodes": {}}
        
        # 관련 노드 정보 가져오기
        related_nodes = self.vector_search.get_related_nodes(first_doc_pmid)
        
        return {
            **state,
            "related_nodes": related_nodes
        }
    
    def _extract_graph_context(self, state: QueryState) -> QueryState:
        """
        그래프 컨텍스트 정보 추출
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # print(f"[디버깅] _extract_graph_context 진입, 전체 state keys: {state.keys()}")
        # print(f"[디버깅] _extract_graph_context 진입, 전체 state['graph_search_output']: {state.get('graph_search_output')}")
        combined_results = state.get("combined_results", [])
        graph_search_output = state.get("graph_search_output", {})
        raw_cypher_results = graph_search_output.get("raw_results")

        # 기본 그래프 컨텍스트 추출 (기존 로직)
        graph_context = {}
        if combined_results:
            graph_context = self.vector_search.get_graph_context_for_response(combined_results)
        
        # Cypher 쿼리 결과(raw_results)가 있으면 graph_context에 추가
        if raw_cypher_results:
            # graph_context가 이미 다른 정보로 채워져 있을 수 있으므로 업데이트 방식 사용
            # 또는 특정 키로 명확히 구분하여 저장 (예: graph_context['cypher_query_results'] = raw_cypher_results)
            # llm.py의 _construct_prompt가 graph_context['raw_results']를 직접 사용하므로, 여기에 맞춤
            graph_context['raw_results'] = raw_cypher_results 
            # print(f"[디버깅] _extract_graph_context: raw_results 추가됨: {raw_cypher_results}")
        # else:
        #     # print("[디버깅] _extract_graph_context: raw_cypher_results 없음.")

        if not graph_context and not combined_results and not raw_cypher_results:
            # print("[디버깅] _extract_graph_context: 사용할 컨텍스트 정보가 없습니다.")

            return {**state, "graph_context": {}}
        
        return {
            **state,
            "graph_context": graph_context
        }
    
    def _generate_response(self, state: QueryState) -> QueryState:
        """
        LLM을 사용하여 최종 응답 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        # print(f"[디버깅] _generate_response 진입, 전체 state keys: {state.keys()}")
        query = state.get("query", "")
        combined_results = state.get("combined_results", [])
        related_nodes = state.get("related_nodes", {})
        graph_context = state.get("graph_context", {})
        # print(f"[디버깅] _generate_response에서 LLM에 전달되는 graph_context: {graph_context}")
        messages = state.get("messages", [])
        
        # 챗 히스토리를 사용하지 않도록 변경
        # chat_history = messages[:-1] if messages else []
        
        # 그래프 검색 결과가 있는지 확인
        graph_search_output = state.get("graph_search_output", {})
        graph_search_results = graph_search_output.get("results", [])
        
        # combined_results가 비어 있지만 graph_search_results가 있으면 이를 사용
        if not combined_results and graph_search_results:
            # print(f"[디버깅] combined_results가 비어 있지만 graph_search_results 사용: {graph_search_results}")
            combined_results = graph_search_results
        
        # 여전히 결과가 없으면 오류 메시지 반환
        if not combined_results:
            # 그래프 검색에서 raw_results만 있는 경우 처리
            raw_results = graph_search_output.get("raw_results", [])
            if raw_results:
                # print(f"[디버깅] combined_results가 없지만 raw_results 사용: {raw_results}")
                # raw_results를 기반으로 가째다 임시 combined_results 생성
                combined_results = [{
                    "pmid": "",
                    "title": "그래프 검색 결과",
                    "abstract": str(raw_results),
                    "search_type": "graph_search"
                }]
            else:
                final_answer = "검색 결과가 없습니다. 다른 질문을 시도해 주세요."
                return {
                    **state,
                    "final_answer": final_answer,
                    "response_quality": "insufficient"  # 결과가 없으면 불충분한 응답으로 표시
                }
        
        # LLM 응답 생성 (챗 히스토리 사용 안함)
        # print("[디버깅] self.llm.generate_response() 호출 직전")
        try:
            final_answer = self.llm.generate_response(
                query=query,
                retrieved_docs=combined_results,
                related_info=related_nodes,
                graph_context=graph_context,
                chat_history=[]  # 빈 배열로 전달하여 히스토리 사용 안함
            )
            # print(f"[디버깅] self.llm.generate_response() 호출 성공, 결과 길이: {len(final_answer)}")
        except Exception as e:
            # print(f"[디버깅] self.llm.generate_response() 호출 중 오류 발생: {e}")
            final_answer = f"검색 결과를 처리하는 중 오류가 발생했습니다: {str(e)}"
        
        
        # 인용 정보 추출
        citations = [
            {
                "pmid": doc.get("pmid"),
                "title": doc.get("title"),
                "search_type": doc.get("search_type", "vector")
            }
            for doc in combined_results[:5]  # 상위 5개 결과만 인용으로 사용
        ]
        
        # 메시지 업데이트
        new_messages = list(messages)
        new_messages.append({"role": "assistant", "content": final_answer})
        
        return {
            **state,
            "final_answer": final_answer,
            "citations": citations,
            "messages": new_messages,
            "response_quality": "unverified",  # 평가 전에는 unverified로 설정
            "tavily_results": [],  # 초기화
            "additional_answer": "",  # 초기화
            "iterations": state.get("iterations", 0)  # 반복 횟수 유지 또는 초기화
        }
    
    def query(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        사용자 쿼리 처리
        
        Args:
            user_query: 사용자 쿼리
            chat_history: 이전 대화 기록
            
        Returns:
            응답 결과
        """
        # 대화 기록이 없으면 빈 리스트로 초기화
        if chat_history is None:
            chat_history = []
        
        # 사용자 메시지 추가
        messages = list(chat_history)
        messages.append({"role": "user", "content": user_query})
        
        # 초기 상태 설정
        initial_state = {
            "messages": messages,
            "query": user_query,
            "query_type": "",
            "vector_results": [],
            "graph_results": [],
            "combined_results": [],
            "related_nodes": {},
            "graph_context": {},
            "final_answer": "",
            "citations": [],
            "response_quality": "unverified",
            "tavily_results": [],
            "additional_answer": "",
            "iterations": 0
        }
        
        # 워크플로우 실행
        result = self.workflow.invoke(initial_state)
        
        # 결과 반환 - 최종 출력 형식 수정
        # messages에는 이미 모든 응답이 포함되어 있음(기본 응답 + 추가 응답)
        response = {
            "answer": result["final_answer"],  # generate_response에서 생성된 원래 응답
            "messages": result["messages"],  # 추가 응답도 포함됨
            "query_type": result["query_type"],
            "retrieved_docs": result["combined_results"],
            "related_info": result["related_nodes"],
            "graph_context": result["graph_context"],
            "citations": result["citations"]
        }
        
        # 추가 응답 정보는 분석 목적으로 전달
        if result.get("additional_answer"):
            response["additional_answer"] = result["additional_answer"]
            response["tavily_results"] = result.get("tavily_results", [])
            print(f"[추가 응답 생성함] {len(result.get('tavily_results', []))} 개의 추가 문서 검색됨")
        
        return response


    def _evaluate_response(self, state: QueryState) -> QueryState:
        """
        LLM 응답의 적절성을 평가
        
        Args:
            state: 현재 상태
            
        Returns:
            평가 결과가 추가된 상태
        """
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        combined_results = state.get("combined_results", [])
        iterations = state.get("iterations", 0)
        query_type = state.get("query_type", "")
        
        # 그래프 검색인 경우 무조건 "good"으로 평가
        if query_type == "graph":
            return {
                **state,
                "response_quality": "good",
                "iterations": iterations
            }
        
        # 반복 횟수 증가 (무한 루프 방지)
        iterations += 1
        if iterations > 2:  # 최대 2회 반복
            return {
                **state,
                "response_quality": "good",  # 강제 종료
                "iterations": iterations
            }
        
        # Tavily API 키가 설정되지 않은 경우 검증 건너뛰기
        if not self.tavily_search.api_key:
            return {
                **state,
                "response_quality": "good",  # API 키가 없으면 그냥 좋은 응답으로 처리
                "iterations": iterations
            }
        
        # 응답이 비어있거나 매우 짧은 경우
        if not final_answer or len(final_answer) < 50:
            return {
                **state,
                "response_quality": "insufficient",
                "iterations": iterations
            }
        
        # LLM을 사용하여 응답의 적절성 평가
        eval_prompt = f"""
        다음 의학 질문과 그에 대한 응답을 분석하세요:
        
        질문: {query}
        
        응답: {final_answer}
        
        이 응답이 다음 기준을 만족하는지 객관적으로 평가해주세요:
        1. 의학적으로 정확한 정보를 제공하는가?
        2. 질문에 충분히 답변했는가?
        3. 추가 정보나 더 많은 의학 연구 결과가 필요한가?
        
        응답 품질을 "good" 또는 "insufficient" 중 하나로 평가해주세요.
        - good: 응답이 충분하고 정확하며 추가 정보가 필요하지 않음

        - insufficient: 응답이 불완전하거나 추가 의학 정보가 필요함

        
        평가 결과만 "good" 또는 "insufficient"로 답변하세요. 다른 설명은 필요하지 않습니다.
        """
        
        try:
            # LLM 호출하여 응답 품질 평가
            quality_assessment = self.llm.model.generate_content(eval_prompt).text.strip().lower()
            
            # 평가 결과 해석 - assessment 자체를 화면에 출력되지 않도록 처리
            if "insufficient" in quality_assessment:
                response_quality = "insufficient"
            else:
                response_quality = "good"
                
            # final_answer는 유지하고 response_quality만 업데이트
            return {
                **state,
                "response_quality": response_quality,
                "iterations": iterations
            }
            
        except Exception as e:
            print(f"응답 평가 중 오류 발생: {e}")
            # 오류 발생 시 무조건 좋은 응답으로 간주 (기본 동작 유지)
            return {
                **state,
                "response_quality": "good",
                "iterations": iterations
            }
    
    def _tavily_search(self, state: QueryState) -> QueryState:
        """
        Tavily API를 사용하여 추가적인 의학 논문 검색
        
        Args:
            state: 현재 상태
            
        Returns:
            Tavily 검색 결과가 추가된 상태
        """
        query = state.get("query", "")
        
        # Tavily 검색 수행
        try:
            tavily_results = self.tavily_search.search_medical_papers(query, max_results=3)
            return {
                **state,
                "tavily_results": tavily_results
            }
        except Exception as e:
            print(f"Tavily 검색 중 오류 발생: {e}")
            return {
                **state,
                "tavily_results": []
            }
    
    def _generate_additional_response(self, state: QueryState) -> QueryState:
        """
        Tavily 검색 결과를 바탕으로 추가 응답 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            추가 응답이 포함된 상태
        """
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        tavily_results = state.get("tavily_results", [])
        messages = state.get("messages", [])
        
        if not tavily_results:
            return state  # 추가 검색 결과가 없으면 기존 상태 유지
        
        # Tavily 검색 결과를 LLM 입력 형식으로 변환
        tavily_context = self.tavily_search.format_results_for_llm(tavily_results)
        
        # 추가 응답 생성을 위한 프롬프트 (프론트엔드에 출력되도록 수정)
        additional_prompt = f"""
        사용자 질문: {query}
                
        다음은 추가로 검색한 의학 논문 정보입니다:
        {tavily_context}
        
        위 추가 정보를 바탕으로 주요 의학적 정보를 포함한 답변을 새롭게 작성해주세요.
        이 답변은 사용자에게 직접 표시됩니다. 
        
        반드시 '추가 정보:'로 시작하여 사용자가 이것이 추가 응답임을 바로 알아볼 수 있도록 하세요.
        가능한 경우 출처(PMID 또는 논문 제목)를 언급해주세요.
        모든 내용은 한국어로 작성하고, 의학적으로 정확한 정보만 제공해야 합니다.
        """
        
        try:
            # LLM 호출하여 추가 응답 생성
            additional_answer = self.llm.model.generate_content(additional_prompt).text
            
            # 추가 인용 정보 추출
            additional_citations = [
                {
                    "pmid": doc.get("pmid"),
                    "title": doc.get("title"),
                    "search_type": "external"
                }
                for doc in tavily_results if doc.get("pmid") or doc.get("title")
            ]
            
            # 기존 인용 정보와 병합
            all_citations = state.get("citations", []) + additional_citations
            
            # 메시지 업데이트 - 추가 응답도 메시지 목록에 삽입하여 화면에 표시되도록 함
            new_messages = list(messages)
            new_messages.append({"role": "assistant", "content": additional_answer})
            
            return {
                **state,
                "additional_answer": additional_answer,
                "citations": all_citations,
                "messages": new_messages
            }
            
        except Exception as e:
            print(f"추가 응답 생성 중 오류 발생: {e}")
            return state  # 오류 발생 시 기존 상태 유지

# 테스트용
if __name__ == "__main__":
    graph_flow = HybridGraphFlow()
    result = graph_flow.query("비만과 고혈압의 연관성에 대해 알려주세요")
    print(f"쿼리 타입: {result['query_type']}")
    print(f"응답: {result['answer']}")
    print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
    if result['retrieved_docs']:
        print(f"첫 번째 문서: {result['retrieved_docs'][0]['title']}")
    if result.get('additional_answer'):
        print(f"\n추가 응답: {result['additional_answer']}")
