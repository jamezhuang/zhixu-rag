from pathlib import Path
from functools import lru_cache
import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    app_mode: str = 'demo'
    database_url: str = 'sqlite:///./data/app.db'
    auth_secret: str = ''
    admin_password: str = ''
    employee_password: str = ''
    deepseek_api_key: str = ''
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: str = 'deepseek-flash'
    deepseek_thinking: bool = False
    deepseek_max_tokens: int = 2400
    structured_output_max_attempts: int = 2
    siliconflow_api_key: str = ''
    siliconflow_base_url: str = 'https://api.siliconflow.cn/v1'
    embedding_model: str = 'Qwen/Qwen3-Embedding-0.6B'
    embedding_dim: int = 1024
    rerank_model: str = 'BAAI/bge-reranker-v2-m3'
    rerank_model_path: str = 'data/models/bge-reranker-v2-m3'
    rerank_device: str = 'cpu'
    rerank_batch_size: int = 4
    rerank_max_length: int = 512
    rerank_cpu_threads: int = 4
    rerank_precision: str = 'int8'
    milvus_uri: str = 'http://localhost:19531'
    milvus_collection: str = 'zhixu_qwen06_1024'
    langsmith_tracing: bool = False
    langsmith_api_key: str = ''
    langsmith_project: str = 'zhixu-rag'
    feishu_app_id: str = ''
    feishu_app_secret: str = ''
    public_app_url: str = 'http://localhost:5173'
    memory_token_budget: int = 4000
    context_token_budget: int = 16000
    context_safety_tokens: int = 800
    context_light_ratio: float = 0.5
    context_medium_ratio: float = 0.7
    context_compact_ratio: float = 0.85
    summary_trigger_ratio: float = 0.7
    recent_rounds: int = 4
    max_retrieval_calls: int = 6
    max_query_retries: int = 2
    request_timeout_seconds: int = 300
    hnsw_ef: int = 64
    enable_fault_injection: bool = False
    data_dir: str = 'data'

    def prepare(self):
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        if not self.auth_secret:
            secret_file = Path(self.data_dir) / '.auth_secret'
            if not secret_file.exists():
                secret_file.write_text(secrets.token_urlsafe(48), encoding='utf-8')
            self.auth_secret = secret_file.read_text(encoding='utf-8').strip()
        if self.app_mode not in ('demo', 'live'):
            raise ValueError('APP_MODE must be demo or live')
        if not 0 < self.summary_trigger_ratio < 1 or self.recent_rounds < 1:
            raise ValueError('Invalid memory configuration')
        if not 0 < self.context_light_ratio < self.context_medium_ratio < self.context_compact_ratio < 1:
            raise ValueError('Context compression thresholds must be increasing ratios in (0, 1)')
        if self.context_safety_tokens < 0 or self.context_token_budget <= self.deepseek_max_tokens + self.context_safety_tokens:
            raise ValueError('Context budget must leave room for input after output and safety reserves')
        if not 0 <= self.max_query_retries <= 10 or self.max_retrieval_calls < 1:
            raise ValueError('MAX_QUERY_RETRIES must be 0..10 and MAX_RETRIEVAL_CALLS must be positive')
        if not 1 <= self.structured_output_max_attempts <= 3:
            raise ValueError('STRUCTURED_OUTPUT_MAX_ATTEMPTS must be 1..3')
        if self.rerank_batch_size < 1 or self.rerank_cpu_threads < 1 or not 64 <= self.rerank_max_length <= 8192:
            raise ValueError('Invalid local reranker configuration')
        if self.rerank_precision not in ('int8', 'fp32', 'fp16'):
            raise ValueError('RERANK_PRECISION must be int8, fp32 or fp16')
        if self.app_mode == 'live' and not self.database_url.startswith('postgresql'):
            raise ValueError('live mode requires PostgreSQL')
        if self.app_mode == 'live' and any(p and len(p) < 8 for p in (self.admin_password, self.employee_password)):
            raise ValueError('初始账号密码至少8位')


@lru_cache
def settings():
    s = Settings()
    s.prepare()
    return s
