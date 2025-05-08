from django.urls import path
from. import views

app_name = 'api' # 앱 네임스페이스 설정 (선택 사항이나 권장)

urlpatterns = [
    path('', views.home, name="homepage"),
    # 문서 정보 API 엔드포인트
    path('document_info/', views.document_info_view, name='document_info'),
    # 향후 AJAX 엔드포인트 등을 위한 경로 추가 가능
    # path('rag_query/', views.rag_query_view, name='rag_query'),
    path('login/', views.login, name="login"),
    path('signup/', views.signup, name="signup"),
    # search/ 경로도 search_view에 매핑
    path('search/', views.search_view, name='search_path'),
    # search/document_info/ 경로 추가
    path('search/document_info/', views.document_info_view, name='search_document_info'),
]
