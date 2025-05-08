from django.db import models
from django.contrib.auth.models import AbstractUser

class UserAccount(AbstractUser):
    groups = None
    user_permissions = None