"""
LangChain 기반 Neo4j 그래프 검색 체인 모듈

LangChain의 GraphCypherQAChain을 사용하여 자연어 질의를 Cypher 쿼리로 변환하고
의학 논문 그래프 데이터베이스에서 검색을 수행합니다.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

class Neo4jGraphChain:
    """
    LangChain 기반 Neo4j 그래프 검색 클래스
    
    사용자 쿼리를 Cypher로 변환하여 의학 논문 그래프에서 검색을 수행합니다.
    """
    
    def __init__(self):
        """Neo4jGraphChain 초기화"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # 필요한 API 키와 연결 정보 확인
        if not self.openai_api_key:
            print("경고: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 API 키를 추가하세요.")
        
        if not self.neo4j_password:
            print("경고: NEO4J_PASSWORD가 설정되지 않았습니다. .env 파일에 암호를 추가하세요.")
        
        # GraphCypherQAChain 초기화
        self._initialize_chain()
    
    def _initialize_chain(self):
        """GraphCypherQAChain 초기화"""
        try:
            # LLM 초기화
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                openai_api_key=self.openai_api_key
            )
            
            # Neo4j URL 디버깅
            print(f"Neo4j 연결 시도: URL={self.neo4j_url}, username={self.neo4j_username}, database={self.neo4j_database}")
            
            # URL이 올바른 형식인지 확인 (bolt:// 또는 neo4j:// 접두사 포함)
            if self.neo4j_url and not (self.neo4j_url.startswith('bolt://') or self.neo4j_url.startswith('neo4j://')):
                self.neo4j_url = f"bolt://{self.neo4j_url}"
                print(f"URL 형식 수정: {self.neo4j_url}")
            
            # Neo4j 그래프 연결
            self.graph = Neo4jGraph(
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                database=self.neo4j_database
            )
            
            # 의학 논문 그래프 스키마 정의 (api/models.py 기반으로 수정)
            self.schema = """
            Nodes:
            - Article: {pmid: STRING, doi: STRING, title: STRING, abstract: STRING, publication_year: INTEGER, publication_date: STRING}
            - Author: {full_name: STRING, last_name: STRING, fore_name: STRING, initials: STRING, affiliation: STRING}
            - Keyword: {term: STRING}
            - Journal: {name: STRING, issn: STRING}

            Relationships:
            - (:Author)-[:AUTHORED_BY]->(:Article)       # Author to Article
            - (:Article)-[:PUBLISHED_IN]->(:Journal)   # Article to Journal
            - (:Article)-[:HAS_KEYWORD]->(:Keyword)     # Article to Keyword
            - (:Article)-[:CITES]->(:Article)           # Optional: if you have citation data
            """
            
            # Cypher 쿼리 생성을 위한 프롬프트 템플릿
            cypher_generation_template = """
            You are an expert Neo4j Developer specializing in medical research papers and graph database queries.
            Your role is to translate user questions into Cypher queries to extract meaningful insights from a medical research papers graph database.

            Instructions:
            1. Use only the provided node labels, relationship types, and properties from the schema.
            2. Do not invent or assume any additional labels, relationships, or properties.
            3. Ensure the query captures the full intent of the user's question while adhering strictly to the schema.
            4. Prioritize efficient graph traversal using Neo4j best practices.
            5. ALL DATA IN THE DATABASE IS IN ENGLISH, so interpret all user queries as searching for English terms, even if the query itself is in Korean.
            6. Always use case-insensitive comparison for string matching (using toLower() function with CONTAINS) for properties like Keyword.term, Article.title, Article.abstract, Author.full_name.
            7. Always limit results (using LIMIT clause) to maximum 15 items.
            8. Return results ordered by relevance when possible (e.g., by publication_year DESC for recent articles).
            9. For queries about articles, always aim to return at least: Article.pmid, Article.title, Article.abstract, Article.publication_year.
            10. For queries about authors, always aim to return at least: Author.full_name, Author.affiliation, and details of their related articles (pmid, title).
            11. If keywords are relevant to the query, return Keyword.term, often collected as a list.
            12. All content in the database is stored in English, so Korean queries should be translated to English for processing.
            Schema: {schema}
            Question: {question}

            Output a Cypher query that accurately answers the user's question based on the provided schema.
            Only return the Cypher query, do not include any explanation.
            """
            
            # 프롬프트 템플릿 생성
            self.cypher_generation_prompt = PromptTemplate(
                template=cypher_generation_template,
                input_variables=["schema", "question"],
            )
            
            # 답변 생성을 위한 프롬프트 템플릿
            self.qa_prompt_template = """
            You are an assistant that takes the results of a Neo4j Cypher query and a question and answers the question.
            Assume the Cypher query has been executed and the results are available.

            The Cypher query results were:
            {context}

            The original question was:
            {question}

            Based on the Cypher query and its results, please answer the original question.
            If the Cypher query results are empty or do not seem to answer the question, say you don't know.
            Only use information from the Cypher query results.
            Always answer in Korean.
            """
            
            self.qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.qa_prompt_template
            )
            
            # GraphCypherQAChain 설정
            self.cypher_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                cypher_prompt=self.cypher_generation_prompt,
                qa_prompt=self.qa_prompt,
                verbose=True,
                return_intermediate_steps=True,  # 중간 단계 결과 반환
                allow_dangerous_requests=True,
                return_direct=True,  # 직접 결과 반환
                top_k=15,  # 최대 15개 결과 반환
            )
            
            print("Neo4jGraphChain 초기화 성공")
            
        except Exception as e:
            print(f"Neo4jGraphChain 초기화 중 오류 발생: {e}")
            self.cypher_chain = None
    
    def search(self, query: str, translate_to_english: bool = True) -> Dict[str, Any]:
        """
        사용자 쿼리를 이용해 그래프 검색 수행
        
        Args:
            query: 자연어 검색 쿼리
            translate_to_english: 쿼리를 영어로 번역할지 여부
            
        Returns:
            검색 결과와 쿼리 정보를 포함한 사전
        """
        if not self.cypher_chain:
            print("GraphCypherQAChain이 초기화되지 않았습니다.")
            return {
                "results": [],
                "error": "GraphCypherQAChain이 초기화되지 않았습니다."
            }
        
        # 원본 쿼리 저장
        original_query = query
        
        # 필요한 경우 쿼리 번역 (여기서는 구현하지 않음, 외부에서 처리 가정)
        english_query = query
        
        try:
            print(f"[디버깅] GraphCypherQAChain으로 쿼리 실행: '{english_query}'")
            
            # 쿼리 실행
            result = self.cypher_chain({"query": english_query})
            
            # 중간 단계 및 생성된 Cypher 쿼리 추출
            intermediate_steps = result.get("intermediate_steps", [])
            cypher_query = intermediate_steps[0]["query"] if intermediate_steps else "쿼리를 생성할 수 없습니다."
            
            # 결과 형식 변환
            formatted_results = []
            raw_results = result.get("result", [])
            
            if isinstance(raw_results, list):
                for item in raw_results:
                    formatted_item = self._format_result_item(item)
                    if formatted_item:
                        formatted_results.append(formatted_item)
            
            print(f"[디버깅] 검색 결과: {len(formatted_results)}건 발견")
            
            # 결과 사전 반환
            return {
                "results": formatted_results,
                "cypher_query": cypher_query,
                "original_query": original_query,
                "english_query": english_query,
                "raw_results": raw_results
            }
            
        except Exception as e:
            print(f"그래프 검색 중 오류 발생: {e}")
            return {
                "results": [],
                "error": str(e),
                "original_query": original_query,
                "english_query": english_query
            }
    
    def _format_result_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        결과 항목을 표준 형식으로 변환
        
        Args:
            item: 원시 결과 항목
            
        Returns:
            표준화된 결과 항목
        """
        try:
            # 핵심 필드 확인
            if not item or not isinstance(item, dict):
                return None
            
            # 필요한 필드 추출
            pmid = item.get("pmid", "")
            title = item.get("title", "제목 없음")
            abstract = item.get("abstract", "초록 없음")
            
            # 키워드 추출 및 표준화
            keywords = item.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            
            # 표준 형식으로 변환
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "search_type": "graph_chain",
                **{k: v for k, v in item.items() if k not in ["pmid", "title", "abstract", "keywords"]}
            }
        except Exception as e:
            print(f"결과 형식 변환 중 오류 발생: {e}")
            return None

# 테스트 코드
if __name__ == "__main__":
    from llm import GeminiLLM # GeminiLLM 임포트를 __main__ 블록 안으로 이동

    chain = Neo4jGraphChain()
    user_test_query = "알츠하이머 를 키워드로 하는 논문의 숫자를 그래프검색으로 알려줘"
    search_result = chain.search(user_test_query)
    
    print("\n==== 생성된 Cypher 쿼리 ====")
    print(search_result.get("cypher_query", "쿼리 없음"))

    print("\n==== 검색 결과 (raw_results) ====")
    raw_results_data = search_result.get("raw_results")
    print(raw_results_data)

    # GeminiLLM을 사용하여 응답 생성 테스트
    if raw_results_data:
        print("\n==== LLM 응답 생성 테스트 ====")
        try:
            gemini_llm = GeminiLLM() # API 키는 GeminiLLM 내부 또는 .env에서 로드됨
            
            # LLM에 전달할 프롬프트를 위한 간단한 컨텍스트 구성
            # 실제 HybridGraphFlow에서는 더 많은 정보가 포함될 수 있음
            llm_query = f"'{user_test_query}'에 대한 그래프 검색 결과를 요약해줘."
            dummy_retrieved_docs = [] # 이 테스트에서는 벡터 검색 문서는 없다고 가정
            graph_context_for_llm = {'raw_results': raw_results_data}
            
            llm_response = gemini_llm.generate_response(
                query=llm_query,
                retrieved_docs=dummy_retrieved_docs, 
                graph_context=graph_context_for_llm
            )
            print("\n==== LLM 생성 응답 ====")
            print(llm_response)
        except Exception as e:
            print(f"LLM 응답 생성 중 오류 발생: {e}")
    else:
        print("\nraw_results가 없어서 LLM 응답 생성 테스트를 건너뜁니다.")
