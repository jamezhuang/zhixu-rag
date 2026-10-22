"""Local cross-encoder inference; no remote rerank endpoint or API key."""
import threading
import time
from pathlib import Path

from langsmith import traceable

from .config import settings
from .providers import call_metrics, call_deadline, trace_inputs


class LocalReranker:
    def __init__(self):
        self.s = settings()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def ready(self):
        folder = Path(self.s.rerank_model_path)
        return ((folder / 'config.json').exists() and any(folder.glob('*.safetensors'))
                and (folder / 'tokenizer_config.json').exists()
                and ((folder / 'tokenizer.json').exists() or (folder / 'sentencepiece.bpe.model').exists()))

    def load(self):
        if self._model is not None:
            return
        if not self.ready():
            raise ValueError('本地重排模型尚未准备。请安装 backend/requirements-reranker.txt 并运行 python -m scripts.prepare_reranker')
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ValueError('缺少本地重排依赖，请安装 backend/requirements-reranker.txt')
        if self.s.rerank_device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('配置了 CUDA，但当前 PyTorch 无可用 CUDA 设备；请使用 RERANK_DEVICE=cpu')
        if self.s.rerank_device == 'cpu':
            torch.set_num_threads(self.s.rerank_cpu_threads)
        path = str(Path(self.s.rerank_model_path).resolve())
        self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        model.to(self.s.rerank_device)
        if self.s.rerank_device.startswith('cuda'):
            model.half()
        model.eval()
        if self.s.rerank_device == 'cpu' and self.s.rerank_precision == 'int8':
            # Dynamic int8 quantizes linear layers; the large token embedding
            # stays float32. In-place replacement avoids a second full model.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        elif self.s.rerank_device == 'cpu' and self.s.rerank_precision == 'fp16':
            raise ValueError('CPU 模式请用 int8 或 fp32；fp16 仅用于 CUDA')
        self._model = model

    @traceable(name='Local BGE Rerank', run_type='chain', process_inputs=trace_inputs)
    def rerank(self, query, candidates, top_n=6):
        if not candidates:
            return []
        # One model instance per process. Serialize inference to bound CPU/RAM
        # consumption; deploy one API worker by default.
        started = time.perf_counter()
        metric = {'endpoint': 'local/rerank', 'model': self.s.rerank_model, 'device': self.s.rerank_device, 'precision': 'fp16' if self.s.rerank_device.startswith('cuda') else self.s.rerank_precision,
                  'candidate_count': len(candidates), 'status': 'error', 'usage': {'api_tokens': 0}}
        try:
            with self._lock:
                self.load()
                scores = []
                for offset in range(0, len(candidates), self.s.rerank_batch_size):
                    deadline = call_deadline.get()
                    if deadline and time.monotonic() >= deadline:
                        raise ValueError('本地重排达到本次问答时间预算，可重试继续')
                    batch = candidates[offset:offset + self.s.rerank_batch_size]
                    pairs = [[query, f'{x["title"]}\n{x["section"]}\n{x["text"]}'] for x in batch]
                    scores.extend(self.score_pairs(pairs))
                if len(scores) != len(candidates):
                    raise ValueError('本地重排分数数量不匹配')
                result = sorted([dict(x, rerank_score=float(score)) for x, score in zip(candidates, scores)],
                                key=lambda x: x['rerank_score'], reverse=True)[:top_n]
                metric['status'] = 'ok'
                return result
        finally:
            metric['elapsed_ms'] = int((time.perf_counter() - started) * 1000)
            collector = call_metrics.get()
            if collector is not None:
                collector.append(metric)

    def score_pairs(self, pairs):
        import torch
        encoded = self._tokenizer(pairs, padding=True, truncation=True, max_length=self.s.rerank_max_length, return_tensors='pt')
        encoded = {k: v.to(self.s.rerank_device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
            return torch.sigmoid(logits).cpu().tolist()
