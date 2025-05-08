from django.urls import path
from. import views

app_name = 'api' # 앱 네임스페이스 설정 (선택 사항이나 권장)

urlpatterns = [
    # 루트 경로('')를 search_view에 매핑하고 이름을 'search'로 지정
    path('', views.search_view, name='search'),
    # search/ 경로도 search_view에 매핑
    path('search/', views.search_view, name='search_path'),
    # 문서 정보 API 엔드포인트
    path('document_info/', views.document_info_view, name='document_info'),
    # 향후 AJAX 엔드포인트 등을 위한 경로 추가 가능
    # path('rag_query/', views.rag_query_view, name='rag_query'),
]
