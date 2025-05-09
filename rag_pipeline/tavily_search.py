"""
의학 논문을 특화된 검색을 위한 Tavily API 통합 모듈

PubMed와 의학 논문 관련 정보에 특화된 검색을 제공합니다.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class TavilySearch:
    """
    Tavily API를 활용한 의학 논문 검색 클래스
    
    PubMed 및 의학 데이터베이스에 특화된 검색을 제공합니다.
    """
    
    def __init__(self):
        """TavilySearch 초기화"""
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("경고: TAVILY_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 추가하세요.")
        
        # Tavily API 클라이언트 초기화
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key) if self.api_key else None
        except ImportError:
            print("Tavily 라이브러리가 설치되지 않았습니다. 'pip install tavily-python' 명령으로 설치하세요.")
            self.client = None
    
    def search_medical_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        의학 논문 검색 수행
        
        Args:
            query: 검색 쿼리
            max_results: 반환할 최대 결과 수
            
        Returns:
            검색 결과 목록. 각 항목에는 제목, 초록, URL, 출처 등이 포함됩니다.
        """
        if not self.client:
            print("Tavily API 클라이언트가 초기화되지 않았습니다.")
            return []
        
        # PubMed 및 의학 정보 사이트에 특화된 검색 수행
        try:
            # 의학 논문 특화 검색어 구성
            medical_query = f"medical research papers about {query} pubmed"
            
            # Tavily 검색 실행
            search_response = self.client.search(
                query=medical_query,
                search_depth="advanced",  # 더 심층적인 검색
                include_domains=["pubmed.ncbi.nlm.nih.gov", "nih.gov", "ncbi.nlm.nih.gov", 
                               "nejm.org", "thelancet.com", "bmj.com", "jamanetwork.com"],
                max_results=max_results,
                include_answer=True,  # Tavily의 요약 응답 포함
                include_raw_content=True  # 원문 콘텐츠 포함
            )
            
            # 결과 형식 변환 및 정제
            results = []
            if 'results' in search_response:
                for item in search_response['results'][:max_results]:
                    # PubMed ID 추출 시도 (없을 수 있음)
                    url = item.get('url', '')
                    pmid = None
                    
                    # PubMed URL에서 PMID 추출 시도
                    if 'pubmed.ncbi.nlm.nih.gov' in url:
                        try:
                            # URL 형식: https://pubmed.ncbi.nlm.nih.gov/PMID/
                            parts = url.rstrip('/').split('/')
                            if len(parts) > 3:
                                potential_pmid = parts[-1]
                                if potential_pmid.isdigit():
                                    pmid = potential_pmid
                        except:
                            pass
                    
                    # 결과 항목 구성
                    result = {
                        'title': item.get('title', '제목 없음'),
                        'content': item.get('content', '내용 없음'),
                        'url': url,
                        'source': 'tavily',
                        'search_type': 'external',
                        'pmid': pmid  # PMID가 추출되지 않았다면 None
                    }
                    results.append(result)
            
            return results
        
        except Exception as e:
            print(f"Tavily 검색 오류: {e}")
            return []
    
    def format_results_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 LLM 입력용으로 포맷팅
        
        Args:
            results: 검색 결과 목록
            
        Returns:
            LLM 입력용으로 포맷팅된 문자열
        """
        if not results:
            return "검색 결과가 없습니다."
        
        formatted_text = "Tavily 검색을 통해 찾은 추가 의학 논문 정보:\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', '제목 없음')
            content = result.get('content', '내용 없음')
            url = result.get('url', '')
            pmid = result.get('pmid', '')
            
            formatted_text += f"문서 {i}:\n"
            formatted_text += f"제목: {title}\n"
            
            # 내용 요약 (너무 길면 잘라냄)
            if len(content) > 500:
                content = content[:500] + "..."
            formatted_text += f"내용: {content}\n"
            
            # 출처 정보
            formatted_text += f"URL: {url}\n"
            if pmid:
                formatted_text += f"PMID: {pmid}\n"
            
            formatted_text += "\n---\n\n"
        
        return formatted_text


# 테스트 코드
if __name__ == "__main__":
    searcher = TavilySearch()
    results = searcher.search_medical_papers("diabetes and cardiovascular disease relationship")
    
    print(f"검색 결과 {len(results)}개 발견:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        if result.get('pmid'):
            print(f"   PMID: {result['pmid']}")
    
    print("\nLLM 입력용 포맷:")
    print(searcher.format_results_for_llm(results[:2]))  # 처음 2개만 표시
