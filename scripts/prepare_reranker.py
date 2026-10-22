"""Download the local model once, then use offline files at runtime."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import httpx


@contextmanager
def download_lock(output):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.download.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit('已有模型准备进程正在运行，请勿重复启动。')
        try:
            yield
        finally:
            lock.seek(0)
            if __import__('os').name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def download_model(model, output):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=15)) as client:
        response = client.get(f'https://huggingface.co/api/models/{model}', params={'blobs': 'true'})
        response.raise_for_status()
        metadata = response.json()
        revision = metadata['sha']
        files = [x for x in metadata['siblings'] if '/' not in x['rfilename'] and x['rfilename'].endswith(('.json', '.safetensors', '.model', '.txt', '.md'))]
        for item in files:
            name = item['rfilename']
            dest = folder / name
            expected_size = item.get('size') or item.get('lfs', {}).get('size')
            expected_sha = item.get('lfs', {}).get('sha256')
            if dest.exists() and (not expected_size or dest.stat().st_size == expected_size):
                if not expected_sha or file_sha(dest) == expected_sha:
                    print('Ready:', name, flush=True)
                    continue
            partial = folder / (name + '.part')
            if expected_size and expected_size > 4 * 1024 * 1024:
                download_ranges(client, f'https://huggingface.co/{model}/resolve/{revision}/{name}', partial, expected_size)
                if expected_sha and file_sha(partial) != expected_sha:
                    raise ValueError('Downloaded model checksum mismatch; remove the .part and .ranges.json files and retry')
                partial.replace(dest)
                partial.with_suffix(partial.suffix + '.ranges.json').unlink(missing_ok=True)
                print('Downloaded and verified:', name, flush=True)
                continue
            for attempt in range(3):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    with client.stream('GET', f'https://huggingface.co/{model}/resolve/{revision}/{name}', headers={'Range': f'bytes={offset}-'} if offset else {}) as stream:
                        stream.raise_for_status()
                        append = stream.status_code == 206 and offset > 0
                        count = offset if append else 0
                        reported = count // (128 * 1024 * 1024)
                        with partial.open('ab' if append else 'wb') as target:
                            for block in stream.iter_bytes(1024 * 1024):
                                target.write(block)
                                count += len(block)
                                if count // (128 * 1024 * 1024) > reported:
                                    reported = count // (128 * 1024 * 1024)
                                    print(f'{name}: {count // (1024 * 1024)} MiB', flush=True)
                    if expected_size and partial.stat().st_size != expected_size:
                        raise ValueError('Downloaded file size mismatch')
                    if expected_sha and file_sha(partial) != expected_sha:
                        raise ValueError('Downloaded model checksum mismatch')
                    partial.replace(dest)
                    print('Downloaded:', name, flush=True)
                    break
                except (httpx.HTTPError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        (folder / '.source.json').write_text(json.dumps({'model': model, 'revision': revision}, indent=2), encoding='utf-8')
    return str(folder.resolve())


def download_ranges(client, url, partial, size):
    """Bounded ranges avoid proxies buffering a multi-GB response before delivery."""
    block_size = 16 * 1024 * 1024
    manifest = partial.with_suffix(partial.suffix + '.ranges.json')
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    valid = previous.get('url') == url and previous.get('size') == size and partial.exists() and partial.stat().st_size == size
    completed = set(previous.get('completed', [])) if valid else set()
    if not valid:
        with partial.open('wb') as stream:
            stream.truncate(size)
    total = (size + block_size - 1) // block_size

    def fetch(index):
        start, end = index * block_size, min((index + 1) * block_size, size) - 1
        for attempt in range(3):
            try:
                response = client.get(url + f'?download=true&range_start={start}', headers={'Range': f'bytes={start}-{end}'})
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get('content-range', '').startswith(f'bytes {start}-{end}/') or len(response.content) != end - start + 1:
                    raise ValueError('Server did not return the requested model range')
                with partial.open('r+b') as target:
                    target.seek(start)
                    target.write(response.content)
                return index
            except (httpx.HTTPError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(fetch, i) for i in range(total) if i not in completed]
        print(f'{partial.name}: {len(completed)}/{total} ranges ready', flush=True)
        for future in as_completed(pending):
            completed.add(future.result())
            manifest.write_text(json.dumps({'url': url, 'size': size, 'completed': sorted(completed)}), encoding='utf-8')
            if len(completed) % 8 == 0 or len(completed) == total:
                print(f'{partial.name}: {len(completed)}/{total} ranges ({min(len(completed) * block_size, size) // (1024 * 1024)} MiB)', flush=True)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--output', default='data/models/bge-reranker-v2-m3')
    parser.add_argument('--verify', action='store_true', help='Run a real CPU inference after downloading')
    args = parser.parse_args()
    with download_lock(args.output):
        path = download_model(args.model, args.output)
    print('Model downloaded:', path)
    if args.verify:
        from backend.app.config import settings
        from backend.app.reranker import LocalReranker
        settings().rerank_model_path = str(Path(path).resolve())
        ranker = LocalReranker()
        result = ranker.rerank('试用期员工可以确认折扣吗？', [
            {'id': 'irrelevant', 'title': '周报', 'section': '提交', 'text': '周五整理经营周报。'},
            {'id': 'relevant', 'title': '试用期权限', 'section': '折扣', 'text': '试用期员工不得独立确认促销折扣，必须经主管审核。'},
        ])
        assert result[0]['id'] == 'relevant', result
        print('Local inference check passed:', [(x['id'], round(x['rerank_score'], 4)) for x in result])


if __name__ == '__main__':
    main()
