import os
import tempfile
from pathlib import Path

import pytest

TEST_DIR = Path(tempfile.mkdtemp(prefix='zhixu-tests-'))
os.environ.update(APP_MODE='demo', DATABASE_URL='sqlite:///' + (TEST_DIR / 'app.db').as_posix(),
                  DATA_DIR=str(TEST_DIR), AUTH_SECRET='test-secret-only-not-used-in-deployment-123456',
                  ENABLE_FAULT_INJECTION='true', LANGSMITH_TRACING='false', ADMIN_PASSWORD='Admin@2026', EMPLOYEE_PASSWORD='Welcome@2026')

from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope='session')
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(client):
    def login(username='operator'):
        response = client.post('/api/auth/login', json={'username': username, 'password': 'Admin@2026' if username == 'admin' else 'Welcome@2026'})
        assert response.status_code == 200, response.text
        return {'Authorization': 'Bearer ' + response.json()['token']}
    return login


@pytest.fixture
def ask(client, auth):
    def send(question, username='operator', conv=None, fault=''):
        headers = auth(username)
        if conv is None:
            conv = client.post('/api/conversations', headers=headers).json()['id']
        created = client.post(f'/api/conversations/{conv}/requests', headers=headers, json={'question': question, 'fault': fault})
        assert created.status_code == 200, created.text
        result = client.post(f'/api/runs/{created.json()["id"]}/execute', headers=headers)
        assert result.status_code == 200, result.text
        return conv, result.json()
    return send
