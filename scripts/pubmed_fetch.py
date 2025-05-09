# scripts/fetch_pubmed.py
from Bio import Entrez
import time
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

#.env 파일에서 환경 변수 로드
load_dotenv()

# NCBI Entrez 사용을 위한 이메일 설정 (필수)
# 실제 이메일 주소로 교체하거나 환경 변수에서 로드
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "your.email@example.com")
if ENTREZ_EMAIL == "your.email@example.com":
    print("경고: 환경 변수 ENTREZ_EMAIL을 설정하거나 코드 내 이메일 주소를 수정하세요.")
Entrez.email = ENTREZ_EMAIL

# API 키 설정 (선택 사항, 호출 빈도 증가)
# ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")
# if ENTREZ_API_KEY:
#     Entrez.api_key = ENTREZ_API_KEY

# 재시도 설정 (기본값: 3회 시도, 15초 간격)
# Entrez.max_tries = 5
# Entrez.sleep_between_tries = 20

print(f"Entrez 이메일 설정됨: {Entrez.email}")

# scripts/fetch_pubmed.py (계속)

def search_recent_pubmed_ids(count=300, days=90):
    """
    지정된 기간 내의 최신 PubMed 논문 ID를 검색합니다.
    Args:
        count (int): 검색할 최대 논문 수.
        days (int): 검색할 최근 일수.
    Returns:
        list: 검색된 PMID 리스트.
    """
    print(f"최근 {days}일 동안의 PubMed 논문 ID {count}개를 검색합니다...")
    try:
        handle = Entrez.esearch(db="pubmed",
                               term="covid 19", # 코로나 
                               retmax=str(count),
                               datetype="pdat", # 출판일 기준
                               reldate=days, # 최근 N일
                               sort="date") # 최신순 정렬 (기본값은 relevance)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]
        print(f"총 {len(pmids)}개의 PMID를 찾았습니다.")
        return pmids
    except Exception as e:
        print(f"PMID 검색 중 오류 발생: {e}")
        return

# 예시: 최근 90일간의 논문 중 최대 300개 ID 검색
pmids_to_fetch = search_recent_pubmed_ids(count=300, days=365)
print(f"가져올 PMID 샘플: {pmids_to_fetch[:5]}...") # 처음 5개 ID 출력

# scripts/fetch_pubmed.py (계속)

def fetch_pubmed_details(pmids, batch_size=100):
    """
    주어진 PMID 리스트에 대해 PubMed 논문 상세 정보를 XML 형식으로 가져옵니다.
    Args:
        pmids (list): 상세 정보를 가져올 PMID 리스트.
        batch_size (int): 한 번의 API 호출로 처리할 PMID 개수.
    Returns:
        list: 각 논문의 상세 정보가 담긴 XML 문자열 리스트.
    """
    all_articles_xml = []
    if not pmids:
        print("가져올 PMID가 없습니다.")
        return all_articles_xml

    print(f"총 {len(pmids)}개 논문의 상세 정보를 가져옵니다 (배치 크기: {batch_size})...")
    for i in range(0, len(pmids), batch_size):
        batch_ids = pmids[i:i + batch_size]
        print(f"배치 {i // batch_size + 1} 처리 중 (PMIDs: {len(batch_ids)}개)...")
        try:
            handle = Entrez.efetch(db="pubmed",
                                  id=batch_ids,
                                  rettype="full", # 또는 'xml'
                                  retmode="xml")
            # XML 데이터를 문자열로 읽기
            xml_data = handle.read()
            handle.close()

            # 개별 PubmedArticle 추출 (ElementTree 사용)
            # Biopython의 Entrez.read/parse는 매우 큰 XML이나 복잡한 구조에서 메모리 문제를 일으킬 수 있음
            # 여기서는 표준 라이브러리 ElementTree를 사용하여 스트리밍 방식으로 처리
            root = ET.fromstring(xml_data)
            articles = root.findall('.//PubmedArticle')
            for article_element in articles:
                # 각 Article을 다시 XML 문자열로 변환하여 저장
                all_articles_xml.append(ET.tostring(article_element, encoding='unicode'))

            print(f"배치 {i // batch_size + 1} 완료. 현재까지 {len(all_articles_xml)}개 논문 처리됨.")
            # NCBI API 호출 제한 준수를 위한 지연 시간 (API 키 없을 시 초당 3회)
            time.sleep(0.4) # 약간의 여유를 둠

        except Exception as e:
            print(f"PMID {batch_ids} 상세 정보 가져오기 중 오류 발생: {e}")
            # 오류 발생 시 다음 배치를 위해 잠시 대기
            time.sleep(5)

    print(f"총 {len(all_articles_xml)}개 논문의 XML 데이터를 성공적으로 가져왔습니다.")
    return all_articles_xml

# 상세 정보 가져오기 실행
articles_xml_list = fetch_pubmed_details(pmids_to_fetch)

# 결과를 파일로 저장 (예: data/pubmed_articles.xml)
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, "pubmed_articles.xml")

# 개별 XML 조각들을 하나의 루트 아래에 합치기
# 각 article XML 문자열이 <?xml...> 헤더를 포함하지 않도록 주의
# ET.tostring()은 기본적으로 헤더 없이 요소만 문자열화 함
combined_root = ET.Element("PubmedArticleSet")
for article_xml in articles_xml_list:
    try:
        article_element = ET.fromstring(article_xml)
        combined_root.append(article_element)
    except ET.ParseError as pe:
        print(f"XML 파싱 오류 (건너뜀): {pe}\nXML 데이터: {article_xml[:200]}...") # 오류 데이터 일부 출력

# 최종 XML 파일 작성
try:
    tree = ET.ElementTree(combined_root)
    # XML 파일 저장 시 encoding='utf-8' 및 xml_declaration=True 명시
    tree.write(output_filepath, encoding='utf-8', xml_declaration=True)
    print(f"총 {len(articles_xml_list)}개 논문 정보가 {output_filepath}에 저장되었습니다.")
except Exception as e:
    print(f"XML 파일 저장 중 오류 발생: {e}")

# 첫 번째 논문 XML 샘플 출력
if articles_xml_list:
    print("\n첫 번째 논문 XML 데이터 샘플:")
    print(str(articles_xml_list)[:500] + "...") # 리스트 전체의 처음 500자 출력# 처음 500자 출력