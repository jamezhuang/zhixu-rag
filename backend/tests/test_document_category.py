from sqlalchemy import select

from backend.app.db import Session, Meta


def test_published_category_update_preserves_policy_and_index(client, auth, monkeypatch):
    headers = auth('admin')
    body = {'code': 'META-001', 'title': 'Category metadata policy', 'version': '1.0',
            'category': '业务流程', 'effective_from': '2026-01-01', 'content': '## Scope\nA policy clause that must remain immutable.'}
    created = client.post('/api/admin/documents', headers=headers, json=body)
    assert created.status_code == 200
    id_ = created.json()['id']
    assert client.post(f'/api/admin/documents/{id_}/publish', headers=headers).status_code == 200
    before = client.get(f'/api/documents/{id_}', headers=headers).json()
    with Session() as db:
        before_revision = db.get(Meta, 'revision').value
    def forbid_reindex(*args):
        raise AssertionError('Catalog metadata must not trigger embedding or reindexing')
    monkeypatch.setattr(client.app.state.retriever, 'index_document', forbid_reindex)
    response = client.patch(f'/api/admin/documents/{id_}/category', headers=headers, json={'category': '入职制度'})
    assert response.status_code == 200
    after = client.get(f'/api/documents/{id_}', headers=headers).json()
    assert after == {**before, 'category': '入职制度'}
    with Session() as db:
        assert db.get(Meta, 'revision').value == before_revision


def test_category_update_checks_permission_and_rejects_content_changes(client, auth):
    docs = client.get('/api/documents', headers=auth('admin')).json()
    id_ = docs[0]['id']
    path = f'/api/admin/documents/{id_}/category'
    assert client.patch(path, json={'category': '公司业务'}).status_code == 401
    assert client.patch(path, headers=auth(), json={'category': '公司业务'}).status_code == 403
    assert client.patch(path, headers=auth('admin'), json={'category': 'unknown'}).status_code == 422
    assert client.patch(path, headers=auth('admin'), json={'category': '公司业务', 'content': 'replacement'}).status_code == 422
    assert client.patch('/api/admin/documents/nonexistent/category', headers=auth('admin'), json={'category': '公司业务'}).status_code == 404
