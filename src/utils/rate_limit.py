"""
Tiện ích chống rate-limit (HTTP 429) cho các provider free-tier.

Gemini free tier giới hạn ~15 request/phút cho LLM và ~100 request/phút cho
embeddings. Bước 3 (RAGAS) gọi LLM ~900 lần và embeddings ~500 lần, nên nếu
không throttle thì chắc chắn bị 429 và cả batch sẽ hỏng giữa chừng.

Module này cung cấp:
    make_rate_limiter()  — InMemoryRateLimiter của LangChain cho ChatModel
    ThrottledEmbeddings  — bọc Embeddings: chia batch + backoff + cache
"""
import os
import time
import pickle
import hashlib
import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.rate_limiters import InMemoryRateLimiter


def make_rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """
    Tạo rate limiter cho ChatModel (LangChain tự chặn trước mỗi lần gọi).

    Args:
        requests_per_minute: số request tối đa mỗi phút

    Returns:
        InMemoryRateLimiter — truyền vào tham số `rate_limiter=` của ChatModel
    """
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=max(1, int(requests_per_minute / 6)),
    )


class ThrottledEmbeddings(Embeddings):
    """
    Bọc một Embeddings instance để:
      1. Chia texts thành batch nhỏ (tránh vượt quota theo phút)
      2. Tự retry với exponential backoff khi gặp lỗi 429 / quota
      3. Cache kết quả theo hash nội dung — cùng một câu hỏi ở Bước 1, 2, 3
         chỉ tốn đúng 1 lần gọi API

    Cache được lưu xuống đĩa (mặc định data/.emb_cache/) nên chạy lại Bước 1/2/3
    gần như không tốn thêm request nào — rất quan trọng với free tier vì
    Gemini chỉ cho 1000 request embeddings mỗi ngày cho mỗi project.

    Dùng như một Embeddings bình thường:
        emb = ThrottledEmbeddings(GoogleGenerativeAIEmbeddings(...), rpm=100)
    """

    def __init__(self, inner: Embeddings, requests_per_minute: int = 100,
                 batch_size: int = 50, max_retries: int = 6,
                 cache_path=None, model_tag: str = ""):
        self._inner        = inner
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._batch_size   = batch_size
        self._max_retries  = max_retries
        self._lock         = threading.Lock()
        self._last_call    = 0.0
        self._model_tag    = model_tag

        # ── cache trên đĩa ────────────────────────────────────────────────
        if cache_path is None:
            root = Path(__file__).parent.parent.parent / "data" / ".emb_cache"
            safe = "".join(c if c.isalnum() else "_" for c in (model_tag or "default"))
            cache_path = root / f"{safe}.pkl"
        self._cache_path = Path(cache_path)
        self._cache      = self._load_cache()
        self._dirty      = 0

    def _load_cache(self) -> dict:
        try:
            if self._cache_path.exists():
                with open(self._cache_path, "rb") as f:
                    data = pickle.load(f)
                print(f"💾 Nạp {len(data)} embedding từ cache {self._cache_path.name}")
                return data
        except Exception as e:
            print(f"⚠️  Không đọc được cache embeddings ({e}) — bắt đầu cache rỗng.")
        return {}

    def _save_cache(self):
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._cache_path)
            self._dirty = 0
        except Exception as e:
            print(f"⚠️  Không ghi được cache embeddings: {e}")

    # ── helpers ────────────────────────────────────────────────────────────
    def _key(self, text: str) -> str:
        # gắn tên model vào key để cache của 2 model khác nhau không lẫn nhau
        return hashlib.sha256(f"{self._model_tag}|{text}".encode("utf-8")).hexdigest()

    def _wait_turn(self, n_requests: int = 1):
        """Giãn cách các lần gọi để không vượt quá quota theo phút."""
        with self._lock:
            now  = time.monotonic()
            gap  = self._min_interval * n_requests
            wait = self._last_call + gap - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _call_with_retry(self, fn, *args, n_requests: int = 1):
        """Gọi hàm embeddings với exponential backoff khi gặp 429/quota."""
        delay = 5.0
        for attempt in range(self._max_retries):
            self._wait_turn(n_requests)
            try:
                return fn(*args)
            except Exception as e:
                msg = str(e).lower()
                is_quota = "429" in msg or "quota" in msg or "rate limit" in msg
                if not is_quota or attempt == self._max_retries - 1:
                    raise
                print(f"    ⏳ Rate limit — chờ {delay:.0f}s rồi thử lại "
                      f"(lần {attempt + 1}/{self._max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 120)

    # ── Embeddings interface ───────────────────────────────────────────────
    def embed_documents(self, texts):
        results  = [None] * len(texts)
        pending  = []          # (index, text) chưa có trong cache

        for i, text in enumerate(texts):
            hit = self._cache.get(self._key(text))
            if hit is not None:
                results[i] = hit
            else:
                pending.append((i, text))

        for start in range(0, len(pending), self._batch_size):
            batch   = pending[start:start + self._batch_size]
            payload = [t for _, t in batch]
            vectors = self._call_with_retry(
                self._inner.embed_documents, payload, n_requests=len(payload)
            )
            for (idx, text), vec in zip(batch, vectors):
                self._cache[self._key(text)] = vec
                results[idx] = vec
                self._dirty += 1

        if self._dirty:
            self._save_cache()
        return results

    def embed_query(self, text: str):
        key = self._key(text)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        vec = self._call_with_retry(self._inner.embed_query, text)
        self._cache[key] = vec
        self._dirty += 1
        self._save_cache()
        return vec

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text: str):
        return self.embed_query(text)
