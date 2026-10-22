import json
import re
import httpx
import time
import asyncio
from contextvars import ContextVar
from langsmith import traceable

from .config import settings
from .compression import compact_payload, SYSTEM_SUFFIX
from .cancellation import cancel_check, check_cancelled

call_metrics = ContextVar('call_metrics', default=None)
call_deadline = ContextVar('call_deadline', default=None)


def trace_inputs(inputs):
    return {k: v for k, v in inputs.items() if k != 'self'}


class ConfigurationError(RuntimeError):
    pass


def parse_json_object(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('模型未返回 JSON 正文，请检查思考模式与输出额度后重试')
    text = content.strip().lstrip('\ufeff').strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S | re.I)
    if fenced:
        text = fenced[1].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Markdown answers may contain literal newlines/tabs inside JSON strings.
        # Preserve their value; do not guess missing quotes/braces or accept other controls.
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            raise ValueError('模型返回的结构化结果无法解析，请重试当前步骤') from None
        try:
            result = json.loads(text, strict=False)
        except json.JSONDecodeError:
            raise ValueError('模型返回的结构化结果无法解析，请重试当前步骤') from None
    if not isinstance(result, dict):
        raise ValueError('模型未返回 JSON 对象')
    return result


def transient(exc):
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or (
        isinstance(exc, httpx.HTTPStatusError) and (exc.response.status_code == 429 or exc.response.status_code >= 500))


class Models:
    def __init__(self):
        self.s = settings()
        self.client = httpx.Client(timeout=httpx.Timeout(60, connect=10))

    def close(self):
        self.client.close()

    async def _cancellable_post(self, url, headers, payload, timeout):
        async with httpx.AsyncClient(timeout=timeout) as client:
            task = asyncio.create_task(client.post(url, headers=headers, json=payload))
            try:
                while not task.done():
                    check_cancelled()
                    await asyncio.wait({task}, timeout=0.2)
                check_cancelled()
                return await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _post(self, base, key, endpoint, payload):
        if not key:
            raise ConfigurationError('模型密钥未配置，请在 .env 补充后重启；或使用 demo 模式')
        started = time.perf_counter()
        deadline = call_deadline.get()
        remaining = deadline - time.monotonic() if deadline else 60
        if remaining <= 0:
            raise ValueError('本次问答已达到执行时间预算，检查点已保留，可重试继续')
        metric = {'endpoint': endpoint, 'model': payload.get('model'), 'status': 'error'}
        try:
            check_cancelled()
            url = f'{base.rstrip("/")}/{endpoint}'
            headers = {'Authorization': f'Bearer {key}'}
            if cancel_check.get():
                response = asyncio.run(self._cancellable_post(url, headers, payload, min(60, remaining)))
            else:
                response = self.client.post(url, headers=headers, json=payload, timeout=min(60, remaining))
            response.raise_for_status()
            result = response.json()
            metric.update(status='ok', usage=result.get('usage', {}))
            return result
        finally:
            metric['elapsed_ms'] = int((time.perf_counter() - started) * 1000)
            collector = call_metrics.get()
            if collector is not None:
                collector.append(metric)

    @traceable(name='DeepSeek JSON', run_type='chain', process_inputs=trace_inputs)
    def json(self, instruction, payload):
        payload, _ = compact_payload(instruction, payload, self.s)
        request = {
            'model': self.s.deepseek_model,
            'messages': [
                {'role': 'system', 'content': instruction + SYSTEM_SUFFIX},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
            ],
            'response_format': {'type': 'json_object'},
            'thinking': {'type': 'enabled' if self.s.deepseek_thinking else 'disabled'},
            'max_tokens': self.s.deepseek_max_tokens,
        }
        for attempt in range(self.s.structured_output_max_attempts):
            data = self._post(self.s.deepseek_base_url, self.s.deepseek_api_key, 'chat/completions', request)
            choice = data['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise ValueError('模型达到输出上限，未完成结构化结果；请关闭思考模式或调整 DEEPSEEK_MAX_TOKENS 后重试')
            collector = call_metrics.get()
            metric = collector[-1] if collector else None
            if metric is not None:
                metric['structured_output_attempt'] = attempt + 1
            try:
                result = parse_json_object(choice.get('message', {}).get('content'))
                if metric is not None:
                    metric['output_format'] = 'ok'
                return result
            except ValueError:
                if metric is not None:
                    metric['output_format'] = 'invalid'
                if attempt + 1 >= self.s.structured_output_max_attempts:
                    raise
                # Retry only this model call, not retrieval or previous graph nodes.
                retry_instruction = instruction + (
                    '\n上次响应格式不合法。重新生成完整 JSON 对象，不加代码围栏或解释。'
                    '字符串内换行必须转义为 \\n，双引号转义为 \\"，确保括号闭合。')
                # The retry instruction also counts toward the complete input budget.
                checked, _ = compact_payload(retry_instruction, payload, self.s)
                request['messages'][0]['content'] = retry_instruction + SYSTEM_SUFFIX
                request['messages'][1]['content'] = json.dumps(checked, ensure_ascii=False)

    @traceable(name='SiliconFlow Embedding', run_type='embedding', process_inputs=trace_inputs)
    def embed(self, texts):
        vectors = []
        for offset in range(0, len(texts), 16):
            batch = texts[offset:offset + 16]
            data = self._post(self.s.siliconflow_base_url, self.s.siliconflow_api_key, 'embeddings', {
                'model': self.s.embedding_model, 'input': batch, 'encoding_format': 'float', 'dimensions': self.s.embedding_dim,
            })
            items = sorted(data['data'], key=lambda x: x['index'])
            if len(items) != len(batch) or any(len(x['embedding']) != self.s.embedding_dim for x in items):
                raise ValueError('Embedding 数量或维度与配置不一致')
            vectors.extend(x['embedding'] for x in items)
        return vectors

