from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# 로컬 모듈 임포트
from rag_pipeline.vector_store import Neo4jVectorSearch
from rag_pipeline.llm import GeminiLLM

# 환경 변수 로드
load_dotenv()

# 상태 클래스 정의
class ChatState(TypedDict):
    messages: List[Dict[str, str]]  # 대화 메시지 목록
    query: str  # 현재 쿼리
    retrieved_docs: List[Dict]  # 검색된 문서
    related_info: Optional[Dict]  # 관련 노드 정보
    connections: Optional[List[Dict]]  # 그래프 연결 정보
    selected_doc_pmid: Optional[str]  # 사용자가 선택한 문서 PMID

class GraphRAG:
    """
    GraphRAG 파이프라인
    
    LangGraph를 사용하여 벡터 검색과 그래프 탐색을 결합한 RAG 시스템 구현
    """
    
    def __init__(self):
        """GraphRAG 초기화"""
        self.vector_search = Neo4jVectorSearch()
        self.llm = GeminiLLM()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        
        # StateGraph 생성
        graph = StateGraph(ChatState)
        
        # 노드 추가
        graph.add_node("retrieve", self._retrieve_documents)
        graph.add_node("extract_related_info", self._extract_related_info)
        graph.add_node("explore_connections", self._explore_connections)
        graph.add_node("generate_response", self._generate_response)
        
        # 엣지 추가 (워크플로우 정의)
        graph.add_edge("retrieve", "extract_related_info")
        graph.add_edge("extract_related_info", "explore_connections")
        graph.add_edge("explore_connections", "generate_response")
        graph.add_edge("generate_response", END)
        
        # 시작 노드 설정
        graph.set_entry_point("retrieve")
        
        # 컴파일 및 반환
        return graph.compile()
    
    def _retrieve_documents(self, state: ChatState) -> ChatState:
        """
        벡터 검색을 통해 관련 문서 검색 후 그래프 기반 리랭킹 적용
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # 벡터 검색 수행 + 그래프 기반 리랭킹 적용
        retrieved_docs = self.vector_search.semantic_search_with_reranking(
            query, 
            initial_k=40,  # 초기 검색 결과 40개
            final_k=5      # 최종 리랭킹 결과 5개 
        )
        
        # 상태 업데이트
        return {
            **state,
            "retrieved_docs": retrieved_docs
        }
    
    def _extract_related_info(self, state: ChatState) -> ChatState:
        """
        문서의 관련 정보 추출 (저자, 키워드, 저널 등)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        retrieved_docs = state.get("retrieved_docs", [])
        
        if not retrieved_docs:
            return {**state, "related_info": None}
        
        # 첫 번째 문서의 PMID로 관련 정보 가져오기
        first_doc_pmid = retrieved_docs[0].get("pmid")
        related_info = self.vector_search.get_related_nodes(first_doc_pmid)
        
        # PMID 추가
        related_info["pmid"] = first_doc_pmid
        
        # 상태 업데이트
        return {
            **state,
            "related_info": related_info,
            "selected_doc_pmid": first_doc_pmid  # 첫 번째 문서 선택
        }
    
    def _explore_connections(self, state: ChatState) -> ChatState:
        """
        그래프 연결 관계 탐색
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        selected_doc_pmid = state.get("selected_doc_pmid")
        
        if not selected_doc_pmid:
            return {**state, "connections": []}
        
        # 그래프 연결 관계 탐색
        connections = self.vector_search.find_article_connections(selected_doc_pmid)
        
        # 상태 업데이트
        return {
            **state,
            "connections": connections
        }
    
    def _generate_response(self, state: ChatState) -> ChatState:
        """
        LLM을 사용하여 응답 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        related_info = state.get("related_info")
        connections = state.get("connections", [])
        messages = state.get("messages", [])
        
        # 챗 히스토리 구성
        chat_history = messages[:-1] if messages else []
        
        # 그래프 컨텍스트 정보 추출
        graph_context = self.vector_search.get_graph_context_for_response(retrieved_docs)
        
        # LLM 응답 생성
        llm_response = self.llm.generate_response(
            query=query,
            retrieved_docs=retrieved_docs,
            related_info=related_info,
            connections=connections,
            chat_history=chat_history,
            graph_context=graph_context  # 그래프 컨텍스트 정보 추가
        )
        
        # 메시지 업데이트
        new_messages = list(messages)
        new_messages.append({"role": "assistant", "content": llm_response})
        
        # 상태 업데이트
        return {
            **state,
            "messages": new_messages
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
            "retrieved_docs": [],
            "related_info": None,
            "connections": [],
            "selected_doc_pmid": None
        }
        
        # 워크플로우 실행
        result = self.workflow.invoke(initial_state)
        
        # 결과 반환
        return {
            "answer": result["messages"][-1]["content"],
            "messages": result["messages"],
            "retrieved_docs": result["retrieved_docs"],
            "related_info": result["related_info"],
            "connections": result["connections"]
        }

# 테스트용
if __name__ == "__main__":
    graph_rag = GraphRAG()
    result = graph_rag.query("비만 치료에 대해 알려주세요")
    print(f"응답: {result['answer']}")
    print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
    if result['retrieved_docs']:
        print(f"첫 번째 문서: {result['retrieved_docs'][0]['title']}") 