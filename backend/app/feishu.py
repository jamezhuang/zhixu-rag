"""Optional private-chat worker. Run only after configuring your own Feishu app."""
import hashlib
import json
import os
import threading

import httpx
import lark_oapi as lark
from lark_oapi.api.im.v1 import ReplyMessageRequest, ReplyMessageRequestBody
from sqlalchemy import select

from .auth import token
from .config import settings
from .db import Session, User, Conversation

API = os.getenv('INTERNAL_API_URL', 'http://127.0.0.1:8000/api')


def process(data):
    message = data.event.message
    if message.chat_type != 'p2p' or message.message_type != 'text':
        return
    open_id = data.event.sender.sender_id.open_id
    text = json.loads(message.content).get('text', '').strip()
    if not text:
        return
    with Session() as db:
        user = db.scalar(select(User).where(User.feishu_open_id == open_id, User.active.is_(True)))
    if not user:
        reply(message.message_id, '请联系管理员，在员工资料中绑定你的飞书 Open ID 后再使用。')
        return
    request_id = hashlib.sha256(message.message_id.encode()).hexdigest()[:32]
    headers = {'Authorization': f'Bearer {token(user)}'}
    with Session.begin() as db:
        conv = db.scalar(select(Conversation).where(Conversation.user_id == user.id, Conversation.channel == 'feishu').order_by(Conversation.created_at.desc()))
        if text == '/new' or not conv:
            conv = Conversation(user_id=user.id, channel='feishu')
            db.add(conv)
            db.flush()
        conv_id = conv.id
    if text == '/new':
        reply(message.message_id, '已开始新的会话，仅带入你的账号资料。')
        return
    with httpx.Client(timeout=600) as client:
        response = client.post(f'{API}/conversations/{conv_id}/requests', headers=headers, json={'question': text, 'request_id': request_id})
        if response.status_code == 409:
            reply(message.message_id, '上一条问题尚未完成，请在网页重试或放弃，或发送 /new 开始新会话。')
            return
        response.raise_for_status()
        # Duplicate events reuse the same request and do not generate another answer.
        if response.json()['status'] == 'completed':
            return
        response = client.post(f'{API}/runs/{request_id}/execute', headers=headers)
        response.raise_for_status()
        result = response.json()
    if result['status'] != 'completed':
        reply(message.message_id, '本次问答未完成，请在网页查看并重试。')
        return
    answer = result['result']['answer']
    sources = result['result']['sources']
    source_text = '\n'.join(f'[{x.get("citation", i + 1)}] {x["title"]} v{x["version"]} · {x["section"]}' for i, x in enumerate(sources[:8]))
    url = f'{settings().public_app_url.rstrip("/")}/?conversation={conv_id}#chat'
    reply(message.message_id, f'{answer}\n\n来源：\n{source_text}\n\n登录网页核对原文：{url}\n发送 /new 开始新会话。')


def reply(message_id, text):
    request = ReplyMessageRequest.builder().message_id(message_id).request_body(
        ReplyMessageRequestBody.builder().content(json.dumps({'text': text[:18000]}, ensure_ascii=False)).msg_type('text').build()).build()
    response = client.im.v1.message.reply(request)
    if not response.success():
        raise RuntimeError(f'Feishu reply failed: {response.code}')


def on_message(data):
    def work():
        try:
            process(data)
        except Exception as exc:
            # Avoid logging message contents, access tokens or app secrets.
            print(f'Feishu event failed: {type(exc).__name__}', flush=True)
    threading.Thread(target=work, daemon=True).start()


if __name__ == '__main__':
    s = settings()
    if not s.feishu_app_id or not s.feishu_app_secret:
        raise SystemExit('请先在 .env 配置 FEISHU_APP_ID 与 FEISHU_APP_SECRET')
    client = lark.Client.builder().app_id(s.feishu_app_id).app_secret(s.feishu_app_secret).build()
    handler = lark.EventDispatcherHandler.builder('', '').register_p2_im_message_receive_v1(on_message).build()
    ws = lark.ws.Client(s.feishu_app_id, s.feishu_app_secret, event_handler=handler, log_level=lark.LogLevel.WARNING)
    ws.start()
