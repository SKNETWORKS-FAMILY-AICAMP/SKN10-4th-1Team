from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
import json
# 새로운 HybridGraphFlow 가져오기
from rag_pipeline.graph_flow import HybridGraphFlow
from rag_pipeline.vector_store import Neo4jVectorSearch

# HybridGraphFlow 인스턴스 생성 (애플리케이션 시작 시 한 번만 초기화)
graph_rag = HybridGraphFlow()
vector_search = Neo4jVectorSearch()

# 초기 검색 페이지 뷰
def search_view(request: HttpRequest) -> HttpResponse:
    """
    검색 페이지를 렌더링합니다.
    POST 요청 시 RAG 파이프라인을 호출하고 결과를 반환합니다.
    """
    context = {}  # 템플릿에 전달할 데이터
    
    if request.method == 'POST':
        try:
            # AJAX 요청에서 데이터 가져오기
            data = json.loads(request.body)
            query = data.get('query', '')
            chat_history = data.get('chat_history', [])
            
            if query:
                # GraphRAG 파이프라인 호출
                result = graph_rag.query(query, chat_history)
                
                # 응답 데이터 구성
                # HybridGraphFlow에서 반환되지 않는 'connections' 키는 빈 리스트로 초기화
                connections = []
                
                # 검색 결과에서 PMID 가져오기
                if result['retrieved_docs'] and len(result['retrieved_docs']) > 0:
                    first_doc = result['retrieved_docs'][0]
                    if 'pmid' in first_doc:
                        # 첫 번째 문서의 연결 관계 가져오기
                        try:
                            connections = vector_search.find_article_connections(first_doc['pmid'])
                        except Exception as e:
                            print(f"쿼리 결과의 연결 관계 가져오기 오류: {e}")
                
                # 기본 응답 데이터 구성
                response_data = {
                    'answer': result['answer'],
                    'messages': result['messages'],  # 주의: messages는 모든 메시지 기록 포함
                    'retrieved_docs': result['retrieved_docs'],
                    'related_info': result['related_info'],
                    'connections': connections,  # 추가로 가져온 연결 관계 혹은 빈 리스트
                    'query_type': result['query_type'], 
                    'graph_context': result.get('graph_context', {}),
                    'citations': result.get('citations', [])
                }
                
                # 추가 응답이 있으면 포함 (Tavily 검색 결과)
                if 'additional_answer' in result and result['additional_answer']:
                    print(f"추가 응답이 생성되었습니다: {len(result['additional_answer'])} 문자")
                    response_data['additional_answer'] = result['additional_answer']
                    if 'tavily_results' in result:
                        response_data['tavily_results'] = result['tavily_results']
                
                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': '검색어를 입력해주세요.'}, status=400)
        
        except Exception as e:
            return JsonResponse({'error': f'오류 발생: {str(e)}'}, status=500)
    
    # GET 요청 시 검색 페이지 렌더링
    return render(request, 'api/chatbot.html', context)

# 선택한 문서의 관련 정보 제공 API
def document_info_view(request: HttpRequest) -> JsonResponse:
    """
    사용자가 선택한 문서의 관련 정보와 연결 관계를 제공합니다.
    """
    if request.method != 'GET':
        return JsonResponse({'error': '잘못된 요청 방식입니다.'}, status=405)
    
    try:
        # URL 파라미터에서 PMID 가져오기
        pmid = request.GET.get('pmid')
        
        if not pmid:
            return JsonResponse({'error': 'PMID가 제공되지 않았습니다.'}, status=400)
        
        # 문서의 관련 정보 가져오기
        related_info = vector_search.get_related_nodes(pmid)
        related_info['pmid'] = pmid  # PMID 추가
        
        # 문서의 연결 관계 가져오기
        connections = vector_search.find_article_connections(pmid)
        
        return JsonResponse({
            'related_info': related_info,
            'connections': connections
        })
    
    except Exception as e:
        print(f"문서 정보 요청 오류: {e}")
        return JsonResponse({'error': f'문서 정보를 가져오는 중 오류가 발생했습니다: {str(e)}'}, status=500)


def home(request) :
    return render(request, 'api/home.html')

def login(request) :
    return render(request, 'api/로그인.html')

def signup(request) :
    return render(request, 'api/회원가입.html')