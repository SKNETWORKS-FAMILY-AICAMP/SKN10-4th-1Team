from django.urls import path
from .views import login_view, signup_view, logout_view

app_name = 'user'  # 앱 네임스페이스 설정 (선택 사항이나 권장)

urlpatterns = [
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
]