import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Gemini API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 GEMINI_API_KEY를 추가해주세요.")

# Gemini AI 설정
genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM:
    """Gemini 2.0 Flash LLM 클라이언트"""
    
    def __init__(self, model_name="gemini-2.0-flash"):
        """
        Gemini LLM 초기화
        
        Args:
            model_name: 사용할 Gemini 모델 이름
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name=model_name)
        
        # 기본 시스템 프롬프트
        self.system_prompt = """
당신은 의학 연구 논문을 검색하고 질문에 답변하는 전문가 AI입니다. 
제공된 논문 정보와 그래프 데이터를 기반으로 정확하고 유용한 답변을 제공하세요.
답변은 항상 한국어로 제공해주세요.

답변 시 반드시 제공된 출처 정보(PMID, 제목, 저자 등)를 인용하고, 각 정보가 어떤 논문에서 나왔는지 명확히 해주세요.
사실 확인이 불가능한 내용은 추측이라고 명시하고, 검색된 정보에서 직접 확인할 수 있는 내용만 포함해주세요.

논문 간의 관계(공통 저자, 키워드 등)가 있다면 이를 강조하여 설명해주세요.
"""
    
    def generate_response(self, 
                         query: str, 
                         retrieved_docs: List[Dict[str, Any]], 
                         related_info: Optional[Dict[str, Any]] = None,
                         connections: Optional[List[Dict[str, Any]]] = None,
                         chat_history: Optional[List[Dict[str, str]]] = None,
                         graph_context: Optional[Dict[str, Any]] = None) -> str:
        """
        LLM을 사용하여 응답 생성
        
        Args:
            query: 사용자 쿼리
            retrieved_docs: 검색된 문서 리스트
            related_info: 관련 정보 (저자, 키워드, 저널 등)
            connections: 그래프 연결 관계
            chat_history: 이전 대화 기록
            graph_context: 그래프 연결 컨텍스트 정보
            
        Returns:
            생성된 응답
        """
        print(f"[디버깅] GeminiLLM.generate_response 진입")
        print(f"[디버깅] GeminiLLM에 전달된 graph_context: {graph_context}")
        if graph_context and 'raw_results' in graph_context:
            print(f"[디버깅] GeminiLLM에 전달된 raw_results: {graph_context['raw_results']}")
        else:
            print("[디버깅] GeminiLLM에 전달된 graph_context에 raw_results가 없습니다.")
        if not retrieved_docs and not graph_context: 
            return "검색된 정보나 그래프 컨텍스트가 없어 답변을 생성할 수 없습니다. 다른 질문을 시도해주세요."
        
        # 프롬프트 구성
        prompt = self._construct_prompt(query, retrieved_docs, related_info, connections, chat_history, graph_context)

        try:
            # 응답 생성
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"응답 생성 오류: {e}")
            return "응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def _construct_prompt(self, 
                         query: str, 
                         retrieved_docs: List[Dict[str, Any]], 
                         related_info: Optional[Dict[str, Any]] = None,
                         connections: Optional[List[Dict[str, Any]]] = None,
                         chat_history: Optional[List[Dict[str, str]]] = None,
                         graph_context: Optional[Dict[str, Any]] = None) -> str:
        """
        LLM 프롬프트 구성
        
        Args:
            query: 사용자 쿼리
            retrieved_docs: 검색된 문서 리스트
            related_info: 관련 정보 (저자, 키워드, 저널 등)
            connections: 그래프 연결 관계
            chat_history: 이전 대화 기록
            graph_context: 그래프 연결 컨텍스트 정보
            
        Returns:
            구성된 프롬프트
        """
        prompt = self.system_prompt + "\n\n"
        
        # 이전 대화 기록 추가
        if chat_history:
            prompt += "### 이전 대화 기록:\n"
            for message in chat_history:
                if message.get("role") == "user":
                    prompt += f"사용자: {message.get('content')}\n"
                else:
                    prompt += f"AI: {message.get('content')}\n"
            prompt += "\n"
        
        # 검색된 문서 정보 추가
        prompt += "### 검색된 논문 정보:\n"
        for i, doc in enumerate(retrieved_docs):
            prompt += f"[논문 {i+1}] PMID: {doc.get('pmid')}\n"
            prompt += f"제목: {doc.get('title')}\n"
            
            if doc.get('abstract'):
                # 초록이 너무 길면 잘라내기
                abstract = doc.get('abstract')
                if len(abstract) > 1000:
                    abstract = abstract[:1000] + "..."
                prompt += f"초록: {abstract}\n"
            
            prompt += f"유사도: {doc.get('similarity', 0):.4f}\n\n"
        
        # 관련 정보 추가 (저자, 키워드, 저널 등)
        if related_info and related_info.get('pmid'):
            prompt += f"### PMID {related_info.get('pmid')}의 관련 정보:\n"
            
            # 저자 정보
            if related_info.get('authors'):
                prompt += "저자:\n"
                for author in related_info.get('authors')[:5]:  # 최대 5명까지
                    prompt += f"- {author.get('full_name')}"
                    if author.get('affiliation'):
                        prompt += f" ({author.get('affiliation')})"
                    prompt += "\n"
                
                if len(related_info.get('authors')) > 5:
                    prompt += f"- 외 {len(related_info.get('authors')) - 5}명\n"
            
            # 키워드 정보
            if related_info.get('keywords'):
                prompt += "키워드: "
                prompt += ", ".join([k.get('term') for k in related_info.get('keywords')])
                prompt += "\n"
            
            # 저널 정보
            if related_info.get('journal'):
                journal = related_info.get('journal')
                prompt += f"저널: {journal.get('name')}"
                if journal.get('issn'):
                    prompt += f" (ISSN: {journal.get('issn')})"
                prompt += "\n"
            
            prompt += "\n"
        
        # 그래프 연결 관계 추가
        if connections:
            prompt += "### 논문 간 연결 관계:\n"
            for i, conn in enumerate(connections[:5]):  # 최대 5개까지
                prompt += f"[연결 {i+1}] 연결 논문: {conn.get('title')}\n"
                
                if conn.get('connecting_node'):
                    node = conn.get('connecting_node')
                    prompt += f"연결 노드: {node.get('type')} - {node.get('name')}\n"
                
                if conn.get('relation_path'):
                    prompt += f"관계: {conn.get('relation_path')}\n"
                
                prompt += "\n"
            
            if len(connections) > 5:
                prompt += f"외 {len(connections) - 5}개 연결 관계가 있습니다.\n\n"
        
        # 그래프 컨텍스트 정보 추가
        if graph_context:
            prompt += "### 그래프 컨텍스트 정보:\n"

            # 그래프 직접 조회 결과 (예: Cypher 쿼리 결과)
            if 'raw_results' in graph_context and graph_context['raw_results']:
                prompt += "그래프 직접 조회 결과:\n"
                results_data = graph_context['raw_results']
                if isinstance(results_data, list):
                    for item_data in results_data:
                        if isinstance(item_data, dict):
                            for key, value in item_data.items():
                                prompt += f"- {key}: {value}\n"
                        else:
                            # 리스트의 항목이 딕셔너리가 아닌 경우 직접 문자열로 변환하여 추가
                            prompt += f"- {str(item_data)}\n"
                elif isinstance(results_data, dict):
                    # 단일 딕셔너리인 경우도 처리
                    for key, value in results_data.items():
                        prompt += f"- {key}: {value}\n"
                else:
                    # 기타 데이터 타입일 경우 문자열로 변환하여 추가
                    prompt += f"- {str(results_data)}\n"
                prompt += "\n"  # 섹션 구분을 위한 줄바꿈

            # 문서 간 연결 관계
            if graph_context.get('document_connections'):
                prompt += "문서 간 연결 관계:\n"
                for i, conn in enumerate(graph_context.get('document_connections')[:3]):
                    source = conn.get('source')
                    target = conn.get('target')
                    connecting_entities = conn.get('connecting_entities', [])
                    relations = conn.get('relations', [])
                    
                    prompt += f"- PMID {source}와 PMID {target} 연결: "
                    if connecting_entities:
                        prompt += f"공통 엔티티: {', '.join(connecting_entities[:2])}"
                    if relations:
                        prompt += f", 관계: {', '.join(relations[:2])}"
                    prompt += "\n"
                
                if len(graph_context.get('document_connections')) > 3:
                    prompt += f"  (외 {len(graph_context.get('document_connections')) - 3}개 연결 관계)\n"
                prompt += "\n"
            
            # 주요 저자 정보
            if graph_context.get('authors'):
                prompt += "주요 저자:\n"
                for name, info in list(graph_context.get('authors').items())[:3]:
                    doc_count = info.get('document_count', 0)
                    prompt += f"- {name}: {doc_count}개 문서 관련\n"
                
                if len(graph_context.get('authors')) > 3:
                    prompt += f"  (외 {len(graph_context.get('authors')) - 3}명)\n"
                prompt += "\n"
            
            # 주요 키워드 정보
            if graph_context.get('keywords'):
                prompt += "주요 키워드:\n"
                for term, info in list(graph_context.get('keywords').items())[:5]:
                    count = info.get('usage_count', 0)
                    related = info.get('related_terms', [])
                    
                    prompt += f"- {term} ({count}회 등장)"
                    if related:
                        prompt += f", 관련: {', '.join(related[:3])}"
                    prompt += "\n"
                
                if len(graph_context.get('keywords')) > 5:
                    prompt += f"  (외 {len(graph_context.get('keywords')) - 5}개 키워드)\n"
                prompt += "\n"
            
            # 관련 토픽 정보
            if graph_context.get('topics'):
                prompt += "관련 토픽:\n"
                for name, info in list(graph_context.get('topics').items())[:3]:
                    doc_count = info.get('document_count', 0)
                    related = info.get('related_topics', [])
                    
                    prompt += f"- {name} ({doc_count}개 문서 관련)"
                    if related:
                        prompt += f", 관련: {', '.join(related[:2])}"
                    prompt += "\n"
                
                if len(graph_context.get('topics')) > 3:
                    prompt += f"  (외 {len(graph_context.get('topics')) - 3}개 토픽)\n"
                prompt += "\n"
        
        # 사용자 쿼리 추가
        prompt += f"### 사용자 질문:\n{query}\n\n"
        prompt += "위 정보를 바탕으로 사용자의 질문에 한국어로 상세히 답변해주세요. 답변에 그래프 연결 관계에서 발견된 흥미로운 패턴이나 관계가 있다면 이를 설명해주세요."
        
        return prompt

# 테스트용
if __name__ == "__main__":
    llm = GeminiLLM()
    test_docs = [
        {
            "pmid": "12345678",
            "title": "Effects of liraglutide on obesity",
            "abstract": "This is a test abstract about obesity treatment with liraglutide.",
            "similarity": 0.89
        }
    ]
    response = llm.generate_response("비만 치료에 대해 알려주세요", test_docs)
    print(response) 