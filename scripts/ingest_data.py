# scripts/ingest_data.py
import xml.etree.ElementTree as ET
from datetime import datetime, date
import os
import sys
from dotenv import load_dotenv
import json  # 명시적 import 추가
import time  # 임베딩 API 호출 제한 방지를 위한 대기 시간 추가

# Django 프로젝트 외부에서 neomodel 모델을 사용하기 위해 경로 추가
# 이 스크립트가 프로젝트 루트에서 실행된다고 가정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Django 설정 로드 (선택 사항, Django 환경 필요 시)
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pubmed_rag.settings')
# import django
# django.setup()

# neomodel 모델 및 설정 로드 (Django 설정 로드 후 또는 직접)
from api.models import Article, Author, Journal, Keyword, install_neomodel_labels
from neomodel import db, DoesNotExist, UniqueProperty, NeomodelException

#.env 파일 로드 및 neomodel 설정 확인
load_dotenv()
NEO4J_BOLT_URL = os.getenv('NEO4J_BOLT_URL')
if not NEO4J_BOLT_URL:
    print("오류:.env 파일에 NEO4J_BOLT_URL 환경 변수를 설정하세요.")
    sys.exit(1)
# neomodel 설정은 api/models.py에서 이미 수행됨

# OpenAI API 설정 (텍스트 임베딩 생성용)
USE_EMBEDDINGS = os.getenv('USE_EMBEDDINGS', 'False').lower() in ('true', '1', 't')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if USE_EMBEDDINGS:
    if not OPENAI_API_KEY:
        print("경고: 임베딩 생성을 활성화했지만 OPENAI_API_KEY가 설정되지 않았습니다.")
        print("임베딩 없이 데이터를 적재합니다.")
        USE_EMBEDDINGS = False
    else:
        try:
            from openai import OpenAI
            print("임베딩 라이브러리 가져오기 성공. 텍스트 임베딩을 생성합니다.")
        except ImportError:
            print("경고: openai 라이브러리를 설치하세요: pip install openai")
            print("임베딩 없이 데이터를 적재합니다.")
            USE_EMBEDDINGS = False

def create_embedding(text, model="text-embedding-3-large", retry_count=3):
    """텍스트 데이터를 OpenAI API를 사용하여 임베딩합니다."""
    if not USE_EMBEDDINGS or not text:
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

def parse_pubmed_xml(xml_filepath):
    """
    PubMed XML 파일을 파싱하여 논문 정보를 딕셔너리 리스트로 반환합니다.
    """
    print(f"'{xml_filepath}' 파일 파싱 시작...")
    articles_data = []
    try:
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        print(f"XML 루트 요소: <{root.tag}>")

        # PubmedArticleSet 아래의 PubmedArticle 요소들을 찾음
        for article_element in root.findall('.//PubmedArticle'):
            article_info = {}
            medline_citation = article_element.find('.//MedlineCitation')
            pubmed_data = article_element.find('.//PubmedData')

            if medline_citation is None:
                print("경고: MedlineCitation 요소를 찾을 수 없습니다. 이 Article을 건너뜁니다.")
                continue

            # PMID 추출
            pmid_element = medline_citation.find('.//PMID')
            article_info['pmid'] = pmid_element.text if pmid_element is not None and pmid_element.text else None
            if not article_info['pmid']:
                print("경고: PMID를 찾을 수 없습니다. 이 Article을 건너뜁니다.")
                continue # PMID는 필수

            # DOI 추출 (PubmedData 아래 ArticleIdList 확인)
            article_info['doi'] = None
            if pubmed_data is not None:
                for article_id in pubmed_data.findall(".//ArticleIdList/ArticleId"):
                    if article_id.text:
                        article_info['doi'] = article_id.text
                        break # 첫 번째 DOI 사용

            # 제목 추출
            title_element = medline_citation.find('.//Article/ArticleTitle')
            article_info['title'] = title_element.text if title_element is not None and title_element.text else "제목 없음"

            # 초록 추출 (복잡할 수 있음, 여러 AbstractText 결합)
            abstract_parts = []
            abstract_element = medline_citation.find('.//Article/Abstract')
            if abstract_element is not None:
                for text_part in abstract_element.findall('.//AbstractText'):
                    # 레이블 속성이 있는 경우 포함 (예: BACKGROUND:, METHODS:)
                    label = text_part.get('Label')
                    text = text_part.text
                    if text:
                        if label:
                            abstract_parts.append(f"{label.strip()}: {text.strip()}")
                        else:
                            abstract_parts.append(text.strip())
            article_info['abstract'] = "\n".join(abstract_parts) if abstract_parts else None

            # 출판일 추출 (연도, 월, 일)
            pub_date_element = medline_citation.find('.//Article/Journal/JournalIssue/PubDate')
            year, month, day = None, None, None
            if pub_date_element is not None:
                year_el = pub_date_element.find('.//Year')
                month_el = pub_date_element.find('.//Month')
                day_el = pub_date_element.find('.//Day')
                if year_el is not None and year_el.text:
                    try: year = int(year_el.text)
                    except ValueError: pass
                if month_el is not None and month_el.text:
                    # 월은 숫자 또는 약어일 수 있음
                    month_str = month_el.text
                    try: month = int(month_str)
                    except ValueError:
                        try: month = datetime.strptime(month_str, '%b').month
                        except ValueError: pass
                if day_el is not None and day_el.text:
                    try: day = int(day_el.text)
                    except ValueError: pass

            article_info['publication_year'] = year
            try:
                if year and month and day:
                    article_info['publication_date'] = date(year, month, day)
                else:
                    article_info['publication_date'] = None
            except ValueError: # 유효하지 않은 날짜 (예: 2월 30일)
                 article_info['publication_date'] = None


            # 저자 정보 추출
            authors = []
            author_list_element = medline_citation.find('.//Article/AuthorList')
            if author_list_element is not None:
                for author_element in author_list_element.findall('.//Author'):
                    last_name_el = author_element.find('.//LastName')
                    fore_name_el = author_element.find('.//ForeName')
                    initials_el = author_element.find('.//Initials')
                    affiliation_el = author_element.find('.//AffiliationInfo/Affiliation') # 첫 번째 소속 정보

                    last_name = last_name_el.text if last_name_el is not None else None
                    fore_name = fore_name_el.text if fore_name_el is not None else None
                    initials = initials_el.text if initials_el is not None else None
                    affiliation = affiliation_el.text if affiliation_el is not None else None

                    if last_name or fore_name: # 이름 정보가 있는 경우만 추가
                        full_name = f"{fore_name or ''} {last_name or ''}".strip()
                        authors.append({
                            'full_name': full_name,
                            'last_name': last_name,
                            'fore_name': fore_name,
                            'initials': initials,
                            'affiliation': affiliation
                        })
            article_info['authors'] = authors

            # 저널 정보 추출
            journal_element = medline_citation.find('.//Article/Journal')
            if journal_element is not None:
                journal_title_el = journal_element.find('.//Title')
                issn_el = journal_element.find('.//ISSN')
                article_info['journal'] = {
                    'name': journal_title_el.text if journal_title_el is not None else None,
                    'issn': issn_el.text if issn_el is not None else None
                }
            else:
                article_info['journal'] = {'name': None, 'issn': None}

            # 키워드 추출 (MeSH Terms)
            keywords = []
            mesh_list_element = medline_citation.find('.//MeshHeadingList')
            if mesh_list_element is not None:
                for mesh_heading in mesh_list_element.findall('.//MeshHeading'):
                    descriptor_name_el = mesh_heading.find('.//DescriptorName')
                    if descriptor_name_el is not None and descriptor_name_el.text:
                        # UI (Unique Identifier) 속성도 포함 가능
                        # ui = descriptor_name_el.get('UI')
                        keywords.append(descriptor_name_el.text)
            # 저자 제공 키워드도 추가 가능 (KeywordList)
            keyword_list_element = medline_citation.find('.//KeywordList')
            if keyword_list_element is not None:
                 for keyword_el in keyword_list_element.findall('.//Keyword'):
                     if keyword_el.text and keyword_el.text not in keywords: # 중복 방지
                         keywords.append(keyword_el.text)

            article_info['keywords'] = keywords

            articles_data.append(article_info)

        print(f"총 {len(articles_data)}개 논문 정보 파싱 완료.")
        return articles_data

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {xml_filepath}")
        return
    except ET.ParseError as e:
        print(f"XML 파싱 오류: {e}")
        return
    except Exception as e:
        print(f"파싱 중 예기치 않은 오류 발생: {e}")
        return

# 날짜 객체를 JSON으로 직렬화하기 위한 사용자 정의 인코더
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()  # ISO 형식 문자열로 변환
        return super().default(obj)

# XML 파일 경로
xml_file = os.path.join("data", "pubmed_articles.xml")
parsed_articles = parse_pubmed_xml(xml_file)

# 파싱된 데이터 샘플 출력
if parsed_articles:
    print("\n첫 번째 파싱된 논문 데이터 샘플:")
    print(json.dumps(parsed_articles[0], indent=2, ensure_ascii=False, cls=DateEncoder))  # 첫 번째 항목만 출력하고 DateEncoder 사용



def ingest_data_to_neo4j(articles_data):
    """
    파싱된 논문 데이터를 neomodel을 사용하여 Neo4j에 적재합니다.
    """
    print(f"\n총 {len(articles_data)}개 논문 데이터 Neo4j 적재 시작...")
    print(f"데이터베이스 연결 URL: {NEO4J_BOLT_URL}")
    print(f"임베딩 생성: {'활성화' if USE_EMBEDDINGS else '비활성화'}")
    success_count = 0
    error_count = 0

    # 스키마 인덱스/제약조건 설치 (최초 실행 시 또는 스키마 변경 시 필요)
    try:
        install_neomodel_labels()
    except Exception as e:
        print(f"스키마 설치 오류 (무시 가능): {e}")

    # 데이터 적재
    for i, article_info in enumerate(articles_data):
        if not article_info.get('pmid'):
            print(f"오류: PMID가 없는 논문 데이터 발견 (인덱스 {i}). 건너뜁니다.")
            error_count += 1
            continue

        try:
            # 트랜잭션 없이 직접 생성 방식으로 시도
            pmid = article_info['pmid']
            title = article_info.get('title', '제목 없음')[:500]  # 제목 길이 제한
            abstract = article_info.get('abstract')
            
            # 임베딩 생성 (USE_EMBEDDINGS가 활성화된 경우)
            title_embedding = None
            abstract_embedding = None
            combined_embedding = None
            
            if USE_EMBEDDINGS:
                # 제목 임베딩
                if title:
                    print(f"PMID {pmid}: 제목 임베딩 생성 중...")
                    title_embedding = create_embedding(title)
                
                # 초록 임베딩
                if abstract:
                    print(f"PMID {pmid}: 초록 임베딩 생성 중...")
                    abstract_embedding = create_embedding(abstract)
                
                # 제목 + 초록 통합 임베딩
                combined_text = ""
                if title:
                    combined_text += title
                if abstract:
                    if combined_text:
                        combined_text += " "
                    combined_text += abstract
                
                if combined_text:
                    print(f"PMID {pmid}: 통합 임베딩 생성 중...")
                    combined_embedding = create_embedding(combined_text)
            
            # 기존 노드가 있는지 확인
            existing_articles = Article.nodes.filter(pmid=pmid)
            if len(existing_articles) > 0:
                print(f"이미 존재하는 Article: PMID {pmid}")
                article_node = existing_articles[0]
                # 업데이트가 필요하다면 여기서 수행
            else:
                # 새 Article 노드 생성
                article_node = Article(
                    pmid=pmid,
                    title=title,
                    abstract=article_info.get('abstract'),
                    doi=article_info.get('doi'),
                    publication_year=article_info.get('publication_year'),
                    publication_date=article_info.get('publication_date'),
                    title_embedding=title_embedding,
                    abstract_embedding=abstract_embedding,
                    combined_embedding=combined_embedding
                )
                article_node.save()
                print(f"Article 생성 성공: PMID {pmid}")
                
                # Journal 연결
                journal_info = article_info.get('journal')
                if journal_info and journal_info.get('name'):
                    try:
                        # 기존 Journal 찾기
                        journal_name = journal_info['name']
                        existing_journals = Journal.nodes.filter(name=journal_name)
                        if len(existing_journals) > 0:
                            journal_node = existing_journals[0]
                        else:
                            # 새로운 Journal 생성
                            journal_node = Journal(
                                name=journal_name,
                                issn=journal_info.get('issn')
                            )
                            journal_node.save()
                        
                        # 연결
                        article_node.journal.connect(journal_node)
                    except Exception as je:
                        print(f"Journal 처리 오류 (PMID {pmid}): {je}")
                
                # Keyword 연결 (각각 별도 try-except으로 처리)
                keywords = article_info.get('keywords', [])
                for term in keywords:
                    if term:  # 빈 키워드 방지
                        try:
                            existing_keywords = Keyword.nodes.filter(term=term)
                            if len(existing_keywords) > 0:
                                keyword_node = existing_keywords[0]
                            else:
                                keyword_node = Keyword(term=term)
                                keyword_node.save()
                            
                            article_node.keywords.connect(keyword_node)
                        except Exception as ke:
                            print(f"Keyword 처리 오류 ({term}, PMID {pmid}): {ke}")
                
                # Author 연결 (각각 별도 try-except으로 처리)
                authors = article_info.get('authors', [])
                for author_info in authors:
                    if author_info.get('full_name'):
                        try:
                            full_name = author_info['full_name']
                            existing_authors = Author.nodes.filter(full_name=full_name)
                            if len(existing_authors) > 0:
                                author_node = existing_authors[0]
                            else:
                                author_node = Author(
                                    full_name=full_name,
                                    last_name=author_info.get('last_name'),
                                    fore_name=author_info.get('fore_name'),
                                    initials=author_info.get('initials'),
                                    affiliation=author_info.get('affiliation')
                                )
                                author_node.save()
                            
                            article_node.authors.connect(author_node)
                        except Exception as ae:
                            print(f"Author 처리 오류 ({author_info.get('full_name')}, PMID {pmid}): {ae}")
            
            success_count += 1
        except UniqueProperty as up:
            print(f"고유 속성 오류 (PMID: {article_info.get('pmid')}): {up}")
            error_count += 1
        except NeomodelException as ne:
            print(f"Neomodel 오류 (PMID: {article_info.get('pmid')}): {ne}")
            error_count += 1
        except Exception as e:
            print(f"처리 중 예기치 않은 오류 발생 (PMID: {article_info.get('pmid')}): {e}")
            error_count += 1
            
        # API 호출 제한 방지를 위한 대기 (임베딩 생성 시)
        if USE_EMBEDDINGS:
            time.sleep(0.5)
            
        # 진행 상황 업데이트
        if (i + 1) % 20 == 0 or i == len(articles_data) - 1:
            print(f"진행 상황: {i + 1}/{len(articles_data)} 논문 처리 완료.")

    print(f"\nNeo4j 데이터 적재 완료.")
    print(f"성공: {success_count}개 논문")
    print(f"오류/건너뜀: {error_count}개 논문")

# 데이터 적재 실행
if parsed_articles:
    ingest_data_to_neo4j(parsed_articles)
else:
    print("파싱된 데이터가 없어 Neo4j 적재를 건너뜁니다.")