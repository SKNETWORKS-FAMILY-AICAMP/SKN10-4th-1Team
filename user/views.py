from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
from user.models import UserAccount 
from django.http import JsonResponse

@csrf_exempt
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')  # 로그인 성공 시 홈으로 리다이렉트
        else:
            print("로그인 실패")
            return render(request, 'user/로그인.html', {'error': 'Invalid credentials'})

    return render(request, 'user/로그인.html')

def signup_view(request):
    if request.method == 'POST':
        full_name = request.POST.get('fullName')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')

        if password != confirm_password:
            return JsonResponse({'error': 'Passwords do not match'}, status=400)

        if UserAccount.objects.filter(email=email).exists():
            return render(request, 'user/회원가입.html', {'error': 'Email already in use'})

        UserAccount.objects.create_user(
            username=email,  # Use email as username
            email=email,
            password=password,
            first_name=full_name
        )
        return redirect('/user/login/')

    return render(request, 'user/회원가입.html')

# Create your views here.
