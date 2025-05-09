from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from user.models import UserAccount 

def logout_view(request):
    if request.method == 'GET':
        logout(request)
        return redirect('api:homepage')  # 로그아웃 후 홈으로 리다이렉트

def login_view(request):
    if request.user.is_authenticated:
        messages.info(request, '이미 로그인되어 있습니다.')
        return redirect('api:homepage')  # 이미 로그인된 경우 홈으로 리다이렉트
    
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('api:homepage')  # 로그인 성공 시 홈으로 리다이렉트
        else:
            messages.error(request, '로그인 실패: 이메일 또는 비밀번호가 잘못되었습니다.')
            return redirect('user:login')

    return render(request, 'user/로그인.html')

def signup_view(request):
    if request.user.is_authenticated:
        messages.info(request, '이미 로그인되어 있습니다.')
        return redirect('api:homepage')  # 이미 로그인된 경우 홈으로 리다이렉트
    
    if request.method == 'POST':
        full_name = request.POST.get('fullName')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')

        if password != confirm_password:
            messages.error(request, '비밀번호가 일치하지 않습니다.')
            return redirect('user:signup')

        if UserAccount.objects.filter(username=email).exists():
            messages.error(request, '이메일이 이미 사용 중입니다.')
            return redirect('user:signup')

        UserAccount.objects.create_user(
            username=email,  # Use email as username
            email=email,
            password=password,
            first_name=full_name
        )
        return redirect('user:login')

    return render(request, 'user/회원가입.html')

# Create your views here.
