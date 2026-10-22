import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .db import Session, User

bearer = HTTPBearer(auto_error=False)


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return f'{salt}:{digest}'


def verify_password(password, encoded):
    salt, digest = encoded.split(':')
    value = hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()
    return hmac.compare_digest(value, digest)


def token(user):
    return jwt.encode({'sub': user.id, 'exp': datetime.now(timezone.utc) + timedelta(hours=12)}, settings().auth_secret, algorithm='HS256')


def current_user(auth: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        data = jwt.decode(auth.credentials if auth else '', settings().auth_secret, algorithms=['HS256'])
        with Session() as db:
            user = db.get(User, data['sub'])
            if not user or not user.active:
                raise ValueError()
            return user
    except (jwt.PyJWTError, ValueError, KeyError):
        raise HTTPException(401, '登录已失效，请重新登录')


def admin(user=Depends(current_user)):
    if user.role != 'admin':
        raise HTTPException(403, '需要管理员权限')
    return user
