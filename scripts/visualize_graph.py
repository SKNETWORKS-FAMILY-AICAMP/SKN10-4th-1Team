import os
import sys
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from py2neo import Graph
import random

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# .env 파일 로드
load_dotenv()

# Neo4j 연결 정보
NEO4J_BOLT_URL = os.getenv('NEO4J_BOLT_URL', '')
username = NEO4J_BOLT_URL.split('://')[1].split(':')[0]
password = NEO4J_BOLT_URL.split(':')[2].split('@')[0]
uri = f"bolt://{NEO4J_BOLT_URL.split('@')[1]}"

def visualize_graph(limit=None, query=None):
    """
    Neo4j 데이터베이스의 그래프를 시각화합니다.
    
    Args:
        limit: 가져올 노드 및 관계의 최대 수 (None이면 제한 없음)
        query: 사용자 정의 쿼리 (기본값은 None, 모든 노드와 관계를 가져옵니다)
    """
    print(f"Neo4j에 연결 중... ({uri})")
    try:
        # Neo4j 연결
        graph = Graph(uri, auth=(username, password))
        print("Neo4j 연결 성공!")
        
        # 쿼리가 없으면 기본 쿼리 사용
        if query is None:
            base_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
            if limit is not None:
                query = f"{base_query} LIMIT {limit}"
            else:
                print("알림: 전체 그래프를 로드합니다. 시간이 오래 걸릴 수 있습니다...")
                query = base_query
        
        print(f"쿼리 실행 중: {query}")
        result = graph.run(query).data()
        print(f"쿼리 결과: {len(result)}개의 결과 반환됨")
        
        if len(result) == 0:
            print("결과가 없습니다.")
            return
        
        # NetworkX 그래프 생성
        G = nx.DiGraph()
        
        # 노드 유형별 색상 매핑
        colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(colors)
        node_types = {}
        node_colors = {}
        
        # 노드와 관계 추가
        for row in result:
            # 시작 노드 속성 확인
            start_node = row['n']
            start_node_id = start_node.identity
            start_node_labels = list(start_node.labels)
            start_node_type = start_node_labels[0] if start_node_labels else "Unknown"
            
            # 대상 노드 속성 확인
            end_node = row['m']
            end_node_id = end_node.identity
            end_node_labels = list(end_node.labels)
            end_node_type = end_node_labels[0] if end_node_labels else "Unknown"
            
            # 관계 속성 확인
            rel = row['r']
            rel_type = type(rel).__name__
            
            # 노드 유형에 색상 할당
            if start_node_type not in node_types:
                node_types[start_node_type] = len(node_types)
                node_colors[start_node_type] = colors[len(node_types) % len(colors)]
            
            if end_node_type not in node_types:
                node_types[end_node_type] = len(node_types)
                node_colors[end_node_type] = colors[len(node_types) % len(colors)]
            
            # 노드 라벨 생성
            start_node_label = f"{start_node_type}"
            for key, value in dict(start_node).items():
                if key in ['name', 'pmid', 'title', 'term', 'full_name'] and value:
                    if key == 'title':
                        # 제목이 너무 길면 잘라냄
                        start_node_label += f"\n{str(value)[:20]}..." if len(str(value)) > 20 else f"\n{value}"
                    else:
                        start_node_label += f"\n{value}"
                    break
            
            end_node_label = f"{end_node_type}"
            for key, value in dict(end_node).items():
                if key in ['name', 'pmid', 'title', 'term', 'full_name'] and value:
                    if key == 'title':
                        # 제목이 너무 길면 잘라냄
                        end_node_label += f"\n{str(value)[:20]}..." if len(str(value)) > 20 else f"\n{value}"
                    else:
                        end_node_label += f"\n{value}"
                    break
            
            # 그래프에 노드 추가
            if not G.has_node(start_node_id):
                G.add_node(start_node_id, 
                          label=start_node_label, 
                          node_type=start_node_type, 
                          color=node_colors[start_node_type])
            
            if not G.has_node(end_node_id):
                G.add_node(end_node_id, 
                          label=end_node_label, 
                          node_type=end_node_type, 
                          color=node_colors[end_node_type])
            
            # 그래프에 엣지 추가
            G.add_edge(start_node_id, end_node_id, label=rel_type)
        
        # 그래프 그리기
        plt.figure(figsize=(15, 10))
        
        # 레이아웃 설정
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 노드 그리기
        for node_type in node_types:
            node_list = [n for n, attr in G.nodes(data=True) if attr['node_type'] == node_type]
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=node_list, 
                                  node_color=[G.nodes[n]['color'] for n in node_list], 
                                  node_size=1000, 
                                  alpha=0.8)
        
        # 노드 라벨 그리기
        nx.draw_networkx_labels(G, pos, 
                               labels={n: attr['label'] for n, attr in G.nodes(data=True)}, 
                               font_size=8, 
                               font_family='sans-serif')
        
        # 엣지 그리기
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
        
        # 엣지 라벨 그리기
        edge_labels = {(u, v): G.edges[u, v]['label'] for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        # 범례 추가
        legend_handles = []
        for node_type in node_types:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=node_colors[node_type], 
                                           markersize=10, label=node_type))
        
        plt.legend(handles=legend_handles, loc='upper right')
        
        # 제목과 여백 설정
        plt.title(f'Neo4j 그래프 시각화 (노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()})', size=15)
        plt.axis('off')
        plt.tight_layout()
        
        # 결과 저장
        output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'graph_visualization.png'), dpi=300)
        
        # 그래프 표시
        plt.show()
        
        print(f"그래프 시각화 완료! 이미지가 {os.path.join(output_path, 'graph_visualization.png')}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")

def visualize_article_with_relations(pmid):
    """특정 논문과 그 관계를 시각화합니다."""
    query = f"""
    MATCH (a:Article {{pmid: '{pmid}'}})-[r]-(m)
    RETURN a as n, r, m
    """
    visualize_graph(query=query)

def visualize_authors_and_articles():
    """저자와 논문 관계를 시각화합니다."""
    query = """
    MATCH (a:Author)-[r:AUTHORED_BY]-(article:Article)
    RETURN a as n, r, article as m
    LIMIT 50
    """
    visualize_graph(query=query)

def visualize_keywords_and_articles():
    """키워드와 논문 관계를 시각화합니다."""
    query = """
    MATCH (k:Keyword)-[r:HAS_KEYWORD]-(article:Article)
    RETURN k as n, r, article as m
    LIMIT 50
    """
    visualize_graph(query=query)

def visualize_journals_and_articles():
    """저널과 논문 관계를 시각화합니다."""
    query = """
    MATCH (j:Journal)-[r:PUBLISHED_IN]-(article:Article)
    RETURN j as n, r, article as m
    LIMIT 50
    """
    visualize_graph(query=query)

if __name__ == "__main__":
    print("Neo4j 그래프 시각화 도구")
    print("-------------------")
    print("1. 전체 그래프 시각화 (제한 없음, 오래 걸릴 수 있음)")
    print("2. 특정 논문과 관계 시각화")
    print("3. 저자와 논문 관계 시각화 (50개 제한)")
    print("4. 키워드와 논문 관계 시각화 (50개 제한)")
    print("5. 저널과 논문 관계 시각화 (50개 제한)")
    print("6. 사용자 정의 쿼리로 시각화")
    
    choice = input("\n선택하세요 (1-6): ")
    
    if choice == '1':
        visualize_graph(limit=None) # limit=None으로 전체 그래프 로드
    elif choice == '2':
        pmid = input("PMID를 입력하세요: ")
        visualize_article_with_relations(pmid)
    elif choice == '3':
        visualize_authors_and_articles() # 기본 limit=50 사용
    elif choice == '4':
        visualize_keywords_and_articles() # 기본 limit=50 사용
    elif choice == '5':
        visualize_journals_and_articles() # 기본 limit=50 사용
    elif choice == '6':
        query = input("Cypher 쿼리를 입력하세요: ")
        custom_limit_str = input("Limit을 지정하시겠습니까? (숫자 또는 Enter): ")
        custom_limit = int(custom_limit_str) if custom_limit_str.isdigit() else None
        visualize_graph(query=query, limit=custom_limit)
    else:
        print("잘못된 선택입니다.") 