#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3-6 -- standalone DashScope API hybrid retrieval evaluation.

Official design: chapter3/retrieval-pipeline/evaluate.py and fusion.py
No repository-local Python imports, no retrieval microservices, no local ML models.

Five genuine stages on identical labeled queries:
  BM25 -> Dense (qwen3.7-text-embedding-flash API) -> Hybrid-RRF ->
  Hybrid-Weighted -> RRF+Rerank (qwen3.7-text-rerank API).

Embedding / rerank are real paid API calls in standard evaluation mode.
Only --self-test simulates HTTP responses; --no-dense is an honest BM25-only baseline.

Doc: https://github.com/bojieli/ai-agent-book/tree/main/chapter3/retrieval-pipeline
Embedding: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
Rerank: https://help.aliyun.com/zh/model-studio/text-rerank-api
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import threading
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from urllib import error as urlerror, request as urlrequest
from urllib.parse import urlparse
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

VERSION = "2.4.0-embedding-index-compat"
UPSTREAM = "https://github.com/bojieli/ai-agent-book/tree/main/chapter3/retrieval-pipeline"
UPSTREAM_EVALUATE_BLOB_SHA = "67de7dfcca766b7dff3bd31c506ee894344fc7cf"
UPSTREAM_FUSION_BLOB_SHA = "1b2fba65b3780d79a33e9058a4fe4d03c1b37a09"

# The 20 documents and 12 labeled questions below reproduce the educational
# DEFAULT_CORPUS and DEFAULT_QUERIES from upstream evaluate.py (Oct 2026).
# No labels are passed into the retrievers: gold is only used in evaluation.
CORPUS = [
    {"doc_id": "xr_7001", "text": "Product model XR-7001 is a smartphone available now."},
    {"doc_id": "xr_7002", "text": "Product model XR-7002 is a smartphone available now."},
    {"doc_id": "xr_7003", "text": "Product model XR-7003 is a smartphone available now."},
    {"doc_id": "xr_7004", "text": "Product model XR-7004 is a smartphone available now."},
    {"doc_id": "xr_7005", "text": "Product model XR-7005 is a smartphone available now."},
    {"doc_id": "xr_7006", "text": "Product model XR-7006 is a smartphone available now."},
    {"doc_id": "http_400", "text": "The HTTP-400 response is a client error status code."},
    {"doc_id": "http_401", "text": "The HTTP-401 response is a client error status code."},
    {"doc_id": "http_403", "text": "The HTTP-403 response is a client error status code."},
    {"doc_id": "http_404", "text": "The HTTP-404 response is a client error status code."},
    {"doc_id": "http_500", "text": "The HTTP-500 response is a server error status code."},
    {"doc_id": "sem_readable", "text": "The language emphasizes clean, readable code that newcomers can pick up quickly."},
    {"doc_id": "sem_gc", "text": "Automatic memory management frees developers from manually releasing objects."},
    {"doc_id": "sem_photo", "text": "Green plants convert sunlight into chemical energy stored as sugars."},
    {"doc_id": "sem_crypto", "text": "Encryption scrambles a message so that only the intended recipient can read it."},
    {"doc_id": "name_alexander_humphrey", "text": "Alexander Humphrey designed the Aurora scheduling protocol in 2019."},
    {"doc_id": "name_alexander_hughes", "text": "Alexander Hughes designed the Borealis scheduling protocol in 2019."},
    {"doc_id": "name_amelia_humphrey", "text": "Amelia Humphrey designed the Celeste scheduling protocol in 2020."},
    {"doc_id": "doc_watercycle", "text": (
        "The water cycle describes how water moves continuously between the ocean, the atmosphere and the land. "
        "Heat from the sun evaporates water from the sea surface into vapor that rises high into the sky. "
        "As the vapor cools it condenses into tiny droplets that gather to form clouds. "
        "When the droplets grow heavy enough they fall back to the ground as rain or snow, "
        "and rivers eventually carry that water back to the ocean, closing the loop."
    )},
    {"doc_id": "doc_volcano", "text": (
        "A volcano forms where molten rock called magma rises from deep inside the planet toward the surface. "
        "Magma collects in a chamber beneath the crust, and mounting pressure forces it upward through cracks. "
        "During an eruption the magma bursts out as lava, ash and gas, which pile up around the vent. "
        "Layer after layer of cooled lava slowly builds the cone-shaped mountain we recognize as a volcano."
    )},
]
QUERIES = [
    {"query": "XR-7003", "expected": ["xr_7003"], "category": "technical-code"},
    {"query": "XR-7005", "expected": ["xr_7005"], "category": "technical-code"},
    {"query": "HTTP-403", "expected": ["http_403"], "category": "technical-code"},
    {"query": "HTTP-400", "expected": ["http_400"], "category": "technical-code"},
    {"query": "Alexander Humphrey", "expected": ["name_alexander_humphrey"], "category": "exact-name"},
    {"query": "a beginner friendly language with tidy syntax", "expected": ["sem_readable"], "category": "semantic"},
    {"query": "reclaiming unused heap space without programmer effort", "expected": ["sem_gc"], "category": "semantic"},
    {"query": "how vegetation turns light into food", "expected": ["sem_photo"], "category": "semantic"},
    {"query": "hiding a note so eavesdroppers cannot understand it", "expected": ["sem_crypto"], "category": "semantic"},
    {"query": "\u690d\u7269\u5982\u4f55\u628a\u9633\u5149\u8f6c\u5316\u4e3a\u98df\u7269", "expected": ["sem_photo"], "category": "multilingual"},
    {"query": "how does water move between the ocean and the sky", "expected": ["doc_watercycle"], "category": "semantic"},
    {"query": "how are volcanoes formed from molten rock", "expected": ["doc_volcano"], "category": "semantic"},
]
METHODS = [
    ("sparse", "BM25"),
    ("dense", "Dense"),
    ("rrf", "Hybrid-RRF"),
    ("weighted", "Hybrid-Weighted"),
    ("rerank", "Hybrid-RRF+Rerank"),
]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0 or not (0 <= overlap < chunk_size):
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    out = []
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size].strip()
        if piece:
            out.append(piece)
        if i + chunk_size >= len(text):
            break
    return out


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
_JIEBA = None
_JIEBA_TRIED = False


def tokenize(text: str) -> List[str]:
    """Keep product IDs intact; optional jieba + character-level CJK recall."""
    global _JIEBA, _JIEBA_TRIED
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        word = m.group()
        if "\u4e00" <= word[0] <= "\u9fff":
            if not _JIEBA_TRIED:
                try:
                    import jieba
                    _JIEBA = jieba
                except ImportError:
                    _JIEBA = None
                _JIEBA_TRIED = True
            if _JIEBA is not None:
                tokens.extend(x for x in _JIEBA.cut(word) if x.strip())
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


class BM25:
    """Pure-Python Okapi BM25 using the rank_bm25 IDF floor convention."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5,
                 b: float = 0.75, epsilon: float = 0.25):
        self.ids = list(ids)
        self.tfs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(freq.values()) for freq in self.tfs]
        n = len(self.tfs)
        self.avgdl = sum(self.lengths) / n if n else 0.0
        df = Counter()
        for termfreq in self.tfs:
            df.update(termfreq.keys())
        original_idf = {
            term: math.log(n - frequency + 0.5) - math.log(frequency + 0.5)
            for term, frequency in df.items()
        }
        mean_idf = (sum(original_idf.values()) / len(original_idf)) if original_idf else 0.0
        floor = epsilon * mean_idf
        self.idf = {t: (v if v >= 0 else floor) for t, v in original_idf.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_terms = tokenize(query)
        result = []
        for doc_id, tf, length in zip(self.ids, self.tfs, self.lengths):
            length_norm = (length / self.avgdl) if self.avgdl else 0.0
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                factor = freq + self.k1 * (1 - self.b + self.b * length_norm)
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / factor
            if score > 0.0:
                result.append((doc_id, score))
        return sorted(result, key=lambda p: (-p[1], p[0]))[:top_k]



EMBED_PATH = "/services/embeddings/text-embedding/text-embedding"
RERANK_PATH = "/services/rerank/text-rerank/text-rerank"


class APIRequestError(RuntimeError):
    """An HTTP API or response-validation failure. NEVER include the API key."""


def _safe_error_message(value: Any) -> str:
    txt = str(value)
    # Avoid echoing accidental API keys if an upstream error embeds one.
    txt = re.sub(r"sk-[A-Za-z0-9_-]{7,}", "sk-[REDACTED]", txt)
    return txt[:550]


def _validated_https_url(url: str, *, purpose: str) -> str:
    """Reject empty/unsafe URLs, credentials, fragments and query strings."""
    value = str(url).strip().rstrip("/")
    parsed = urlparse(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError(f"{purpose} must be an absolute HTTPS URL without credentials/query/fragment")
    if ("<" in value or ">" in value or "{" in value or "}" in value):
        raise ValueError(f"{purpose} contains a placeholder; supply your real address in .env")
    return value


def _validated_api_base(value: str) -> str:
    value = _validated_https_url(value, purpose="DashScope API base URL")
    if not value.endswith("/api/v1"):
        raise ValueError("Native DashScope API base URL must end with /api/v1")
    return value


def _native_base(value: str) -> str:
    """Normalize DashScope compatible-mode / legacy host to its native /api/v1.

    Alibaba documents this path switch for both shared and workspace hosts.
    Never transform a third-party OpenAI-compatible URL into a DashScope one.
    """
    value = _validated_https_url(value, purpose="DashScope API URL")
    for suffix in (EMBED_PATH, RERANK_PATH):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    for suffix in ("/compatible-mode/v1", "/compatible-api/v1"):
        if value.endswith(suffix):
            value = value[:-len(suffix)] + "/api/v1"
    if value.endswith("/api/v1"):
        return _validated_api_base(value)
    parsed = urlparse(value)
    if parsed.path in ("", "/") and parsed.hostname and parsed.hostname.endswith("aliyuncs.com"):
        return _validated_api_base(value + "/api/v1")
    raise ValueError(
        "Unrecognized DashScope URL path. Set DASHSCOPE_HTTP_BASE_URL=.../api/v1 "
        "or use an Alibaba Cloud .../compatible-mode/v1 URL. "
        "A generic third-party /v1 endpoint cannot serve the native Embedding/Rerank protocol."
    )


def _env_first(*names: str) -> Tuple[str, str]:
    """Return first non-empty environment value and the variable name used."""
    for name in dict.fromkeys(names):
        if not name:
            continue
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return "", ""


def _load_dotenv(paths: Sequence[Path], *, override: bool = False) -> Dict[str, Any]:
    """Load project .env using stdlib only. Never log or persist secret values.

    Priority is first .env file, then later files. Existing non-empty shell env
    wins by default; --env-override enables .env to replace shell values.
    """
    seen = set()
    loaded_files = []
    found_keys = set()
    applied_keys = set()
    for original in paths:
        try:
            path = Path(original).expanduser().resolve()
        except OSError:
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        loaded_files.append(str(path))
        try:
            raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Failed reading .env file {path}: {type(exc).__name__}") from exc
        for raw in raw_lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            name, equal, value = line.partition("=")
            name = name.strip()
            if not equal or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue
            value = value.strip()
            if value.startswith(("\"", "'")):
                # Handles quoted values followed by comments; shlex does not
                # expose the secret in errors or debug logs.
                import shlex
                try:
                    parts = shlex.split(value, comments=True, posix=True)
                except ValueError as exc:
                    raise ValueError(f"Malformed quoted value in .env: {path}") from exc
                if len(parts) != 1:
                    raise ValueError(f"Malformed quoted value in .env: {path}")
                value = parts[0]
            else:
                value = value.split(" #", 1)[0].strip()
            found_keys.add(name)
            if (name not in os.environ or not os.environ[name].strip() or override) and name not in applied_keys:
                os.environ[name] = value
                applied_keys.add(name)
    return {"files": loaded_files, "keys_detected": sorted(found_keys),
            "keys_applied": sorted(applied_keys)}


def _env_paths(config_path: Path, env_file: Optional[str], *, required: bool = False) -> List[Path]:
    """Configured .env is always searched first, regardless of invocation CWD."""
    if env_file:
        target = Path(env_file).expanduser()
        if not target.is_absolute():
            target = config_path.parent / target
        if not target.is_file():
            if required:
                raise FileNotFoundError(f"Explicit .env file not found: {target}")
            return []
        return [target]
    bases = [config_path.parent, Path.cwd(), Path(__file__).resolve().parent]
    bases.extend(Path(__file__).resolve().parents[:6])
    candidates = [config_path.parent / '.env']
    candidates.extend(base / '.env' for base in bases)
    return list(dict.fromkeys(candidates))


def _configured_env_names(cfg: Mapping[str, Any], key: str, *aliases: str) -> List[str]:
    configured = str(cfg.get(key) or "").strip()
    return list(dict.fromkeys(x for x in (configured, *aliases) if x))


def _resolve_full_endpoint(value: str, stage: str) -> str:
    """Accept stage-specific native endpoint, or native/compatible base URL."""
    value = _validated_https_url(value, purpose=f"{stage} API URL")
    endpoint = EMBED_PATH if stage == "embedding" else RERANK_PATH
    wrong = RERANK_PATH if stage == "embedding" else EMBED_PATH
    if value.endswith(wrong):
        raise ValueError(f"{stage} API URL points to the wrong endpoint")
    if value.endswith(endpoint):
        return value
    return _native_base(value) + endpoint


def resolve_api_endpoints(api: Mapping[str, Any], overrides: Any) -> Dict[str, str]:
    """Resolve .env aliases, CLI and YAML into working native HTTP endpoints.

    Priority: CLI override > preferred env names > legacy env aliases > YAML
    host > explicit workspace > official shared native host by region.
    Full stage endpoints can be supplied independently, with no Workspace ID.
    """
    region = (getattr(overrides, "region", None) or
              _env_first(*_configured_env_names(api, "region_env", "DASHSCOPE_REGION", "ALIYUN_REGION"))[0]
              or str(api.get("region") or "cn-beijing").strip())
    official_shared_hosts = {
        "cn-beijing": "https://dashscope.aliyuncs.com/api/v1",
        "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/api/v1",
        "us-east-1": "https://dashscope-us.aliyuncs.com/api/v1",
    }
    workspace, workspace_source = _env_first(*_configured_env_names(
        api, "workspace_id_env", "DASHSCOPE_WORKSPACE_ID", "BAILIAN_WORKSPACE_ID", "WORKSPACE_ID"))
    workspace = (getattr(overrides, "workspace_id", None) or workspace
                 or str(api.get("workspace_id") or "").strip())
    if getattr(overrides, "workspace_id", None):
        workspace_source = "CLI --workspace-id"
    elif workspace and not workspace_source:
        workspace_source = "YAML api.workspace_id"
    if workspace and not re.fullmatch(r"[A-Za-z0-9-]+", workspace):
        raise ValueError("Workspace ID may contain only ASCII letters, digits and hyphens")

    # Any explicit URL in .env is authoritative. Do not try to infer its provider.
    embed_raw, embed_source = _env_first(*_configured_env_names(
        api, "embedding_url_env", "DASHSCOPE_EMBEDDING_URL", "DASHSCOPE_EMBEDDING_API_URL",
        "QWEN_EMBEDDING_URL", "EMBEDDING_API_URL", "EMBEDDING_URL"))
    rerank_raw, rerank_source = _env_first(*_configured_env_names(
        api, "rerank_url_env", "DASHSCOPE_RERANK_URL", "DASHSCOPE_RERANKER_URL",
        "QWEN_RERANK_URL", "RERANKER_API_URL", "RERANK_API_URL", "RERANK_URL"))
    if not embed_raw and api.get("embedding_url"):
        embed_raw, embed_source = str(api["embedding_url"]), "YAML api.embedding_url"
    if not rerank_raw and api.get("rerank_url"):
        rerank_raw, rerank_source = str(api["rerank_url"]), "YAML api.rerank_url"
    embed_url = _resolve_full_endpoint(embed_raw, "embedding") if embed_raw else ""
    rerank_url = _resolve_full_endpoint(rerank_raw, "rerank") if rerank_raw else ""

    base_raw = getattr(overrides, "api_base_url", None)
    base_source = "CLI --api-base-url" if base_raw else ""
    if not base_raw and getattr(overrides, "workspace_id", None):
        # The explicit CLI workspace must beat an inherited legacy base URL.
        base_raw = f"https://{workspace}.{region}.maas.aliyuncs.com/api/v1"
        base_source = "CLI --workspace-id"
    if not base_raw:
        base_raw, base_source = _env_first(*_configured_env_names(
            api, "base_url_env", "DASHSCOPE_HTTP_BASE_URL", "DASHSCOPE_NATIVE_BASE_URL",
            "DASHSCOPE_API_BASE_URL", "DASHSCOPE_API_URL", "DASHSCOPE_BASE_URL",
            "QWEN_API_BASE_URL", "QWEN_API_URL", "QWEN_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"))
        if base_raw and base_source in ("OPENAI_BASE_URL", "BASE_URL"):
            # Ignore generic variables from another provider, rather than
            # directing native DashScope requests to an unrelated service.
            host = urlparse(base_raw).hostname or ""
            if not host.endswith("aliyuncs.com"):
                base_raw, base_source = "", ""
    if not base_raw:
        base_raw = str(api.get("http_base_url") or api.get("base_url") or "").strip()
        if base_raw:
            base_source = "YAML api.http_base_url"
    if base_raw:
        base = _native_base(base_raw)
    elif workspace:
        if not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+", region):
            raise ValueError(f"Invalid region: {region!r}")
        base = _validated_api_base(f"https://{workspace}.{region}.maas.aliyuncs.com/api/v1")
        base_source = workspace_source
    elif embed_url or rerank_url:
        first = embed_url or rerank_url
        suffix = EMBED_PATH if embed_url else RERANK_PATH
        base = _validated_api_base(first[:-len(suffix)])
        base_source = embed_source or rerank_source
    elif region in official_shared_hosts:
        # Documented native shared hosts are valid without Workspace ID.
        # Account/key region must still match; real requests may return 401.
        base = official_shared_hosts[region]
        base_source = f"official shared host ({region})"
    else:
        raise ValueError(
            f"No API URL or Workspace ID, and no shared endpoint for region {region!r}. "
            "Set DASHSCOPE_HTTP_BASE_URL or DASHSCOPE_WORKSPACE_ID in .env."
        )
    embed_url = embed_url or base + EMBED_PATH
    rerank_url = rerank_url or base + RERANK_PATH
    return {
        "base_url": base, "embedding_url": embed_url, "rerank_url": rerank_url,
        "base_source": base_source, "embedding_source": embed_source or base_source,
        "rerank_source": rerank_source or base_source, "region": region,
    }


def resolve_api_base(api: Mapping[str, Any], overrides: Any) -> str:
    """Backwards-compatible helper used by some external tests."""
    return resolve_api_endpoints(api, overrides)["base_url"]


class DashScopeHTTP:
    """Use DashScope native endpoints; checks official response shapes and receipts."""

    def __init__(self, base_url: str, key: str, *, timeout_sec: float = 60.0,
                 max_retries: int = 3, retry_base_sec: float = 1.5,
                 max_http_requests: int = 100, embedding_url: str = "",
                 rerank_url: str = ""):
        if not key.strip():
            raise ValueError("Missing DASHSCOPE_API_KEY; export it or add to root .env")
        if not 1 <= timeout_sec <= 300:
            raise ValueError("timeout_sec must be between 1 and 300")
        if not 0 <= max_retries <= 10 or not 0 <= retry_base_sec <= 30:
            raise ValueError("Invalid retries/backoff")
        if max_http_requests < 1:
            raise ValueError("max_http_requests must be positive")
        self.base_url = _validated_api_base(base_url)
        self.url_embedding = embedding_url.strip() or self.base_url + EMBED_PATH
        self.url_rerank = rerank_url.strip() or self.base_url + RERANK_PATH
        for candidate in (self.url_embedding, self.url_rerank):
            u = urlparse(candidate)
            if u.scheme != "https" or not u.netloc or u.username or u.password or u.query or u.fragment:
                raise ValueError("API URLs must be absolute HTTPS URLs without credentials or query")
        self._key = key
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max_retries
        self.retry_base_sec = float(retry_base_sec)
        self.max_http_requests = max_http_requests
        self.records: List[Dict[str, Any]] = []
        self.attempted_http_requests = 0
        self.lock = threading.Lock()

    def _reserve_request(self) -> None:
        with self.lock:
            if self.attempted_http_requests >= self.max_http_requests:
                raise APIRequestError(f"API request budget exceeded: {self.max_http_requests}")
            self.attempted_http_requests += 1

    def post(self, stage: str, payload: Mapping[str, Any], count: int) -> Dict[str, Any]:
        if stage not in ("embedding", "rerank"):
            raise ValueError(f"Invalid API stage: {stage}")
        url = self.url_embedding if stage == "embedding" else self.url_rerank
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_problem = "unknown"
        started = now_ms()
        for attempt in range(self.max_retries + 1):
            self._reserve_request()
            req = urlrequest.Request(url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ai-agent-book-exp3-6-standalone-api/2.0",
            })
            retry = False
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_sec) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                    status = getattr(response, "status", 200)
                    if status >= 400:
                        raise APIRequestError(f"HTTP {status}: {parsed.get('message', '')}")
            except urlerror.HTTPError as exc:
                error_body = exc.read(4096).decode("utf-8", errors="replace")
                try:
                    remote = json.loads(error_body)
                    problem = remote.get("message") or remote.get("error") or remote.get("code") or error_body
                except ValueError:
                    problem = error_body
                last_problem = f"HTTP {exc.code}: {_safe_error_message(problem)}"
                retry = exc.code in (408, 409, 429, 500, 502, 503, 504)
            except (urlerror.URLError, TimeoutError, ConnectionError) as exc:
                last_problem = _safe_error_message(exc)
                retry = True
            except ValueError as exc:
                raise APIRequestError(f"{stage} API returned invalid JSON: {_safe_error_message(exc)}") from exc
            else:
                if not isinstance(parsed, dict):
                    raise APIRequestError(f"{stage} API returned a non-object JSON value")
                if parsed.get("code") or parsed.get("error"):
                    raise APIRequestError(f"{stage} API rejected request: {_safe_error_message(parsed.get('message') or parsed.get('error') or parsed.get('code'))}")
                if parsed.get("status_code") not in (None, 200):
                    raise APIRequestError(f"{stage} API status_code={parsed['status_code']}: {_safe_error_message(parsed.get('message', 'unknown error'))}")
                usage = parsed.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                receipt = {
                    "stage": stage,
                    "model": payload.get("model"),
                    "items": count,
                    "request_id": parsed.get("request_id") or parsed.get("id"),
                    "usage": {k: usage[k] for k in ("prompt_tokens", "total_tokens") if k in usage},
                    "elapsed_ms": round(now_ms() - started, 2),
                    "attempts": attempt + 1,
                }
                with self.lock:
                    self.records.append(receipt)
                return parsed
            if not retry or attempt == self.max_retries:
                raise APIRequestError(f"{stage} API failed: {last_problem}. Verify WorkspaceId, region and model permissions.")
            time.sleep(min(30.0, self.retry_base_sec * (2 ** attempt)))
        raise APIRequestError(f"{stage} API failed: {last_problem}")

    def summary(self) -> Dict[str, Any]:
        with self.lock:
            receipts = list(self.records)
            attempts = self.attempted_http_requests
        return {
            "http_requests_attempted": attempts,
            "successful_calls": len(receipts),
            "calls_by_stage": dict(Counter(x["stage"] for x in receipts)),
            "total_tokens_reported": sum(int(x["usage"].get("total_tokens", 0)) for x in receipts),
            "receipts": receipts,
        }


def _unit_vector(values: Sequence[float], expected_dimension: int) -> Tuple[float, ...]:
    if not isinstance(values, list) or len(values) != expected_dimension:
        raise APIRequestError(f"Embedding dimension mismatch: expected {expected_dimension}, got {len(values) if isinstance(values, list) else 'not a list'}")
    numbers = [float(x) for x in values]
    if any(not math.isfinite(x) for x in numbers):
        raise APIRequestError("Embedding contained NaN or infinity")
    length = math.sqrt(sum(v * v for v in numbers))
    if not length:
        raise APIRequestError("Embedding returned zero vector")
    return tuple(v / length for v in numbers)


def _embedding_index(item: Mapping[str, Any], row: int, size: int,
                     request_id: str) -> int:
    """Safely resolve native text_index and OpenAI-style index if supplied.

    Missing indices are only unambiguous for a one-text request. In a batch we
    MUST NOT infer ordering, or retrieval labels could be silently corrupted.
    """
    candidates = []
    for field in ("text_index", "index"):
        if field not in item or item[field] is None:
            continue
        raw = item[field]
        if type(raw) is int:
            idx = raw
        elif isinstance(raw, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)", raw):
            idx = int(raw)
        else:
            raise APIRequestError(
                f"Embedding {field} has invalid type {type(raw).__name__} "
                f"in row {row}; request_id={request_id}"
            )
        candidates.append((field, idx))
    if not candidates:
        if size == 1:
            return 0
        keys = sorted(str(x) for x in item.keys())
        raise APIRequestError(
            f"Embedding response row {row} has no text_index/index for a "
            f"{size}-text batch; item keys={keys}; request_id={request_id}. "
            "Cannot safely align vectors to input texts. "
            "Use --batch-size 1 to avoid ambiguous alignment, or inspect the "
            "provider's response schema."
        )
    idx = candidates[0][1]
    if any(value != idx for _, value in candidates[1:]):
        raise APIRequestError(
            f"Embedding text_index and index disagree in row {row}; "
            f"request_id={request_id}"
        )
    if not 0 <= idx < size:
        raise APIRequestError(
            f"Embedding index out of range in row {row} for batch size {size}; "
            f"request_id={request_id}"
        )
    return idx


def _read_embedding_response(response: Mapping[str, Any], size: int,
                             dimension: int) -> List[Tuple[float, ...]]:
    """Parse actual provider results; never fabricate missing vectors or order.

    Native: output.embeddings with text_index; some gateways use index.
    Compatible: data with index. A singleton without index maps to the only
    input text, but a multi-text response without indices is rejected.
    """
    output = response.get("output")
    native = output.get("embeddings") if isinstance(output, dict) else None
    compatible = response.get("data")
    if isinstance(native, list):
        items = native
    elif isinstance(compatible, list):
        items = compatible
    else:
        raise APIRequestError(
            "Embedding response has no output.embeddings/data array; "
            f"top-level keys={sorted(str(k) for k in response.keys())}; "
            f"request_id={response.get('request_id') or response.get('id') or 'n/a'}"
        )
    if len(items) != size:
        raise APIRequestError(
            f"Embedding returned {len(items)} vectors for {size} input texts; "
            f"request_id={response.get('request_id') or response.get('id') or 'n/a'}"
        )
    request_id = str(response.get("request_id") or response.get("id") or "n/a")
    ordered: List[Optional[Tuple[float, ...]]] = [None] * size
    for row, item in enumerate(items):
        if not isinstance(item, dict):
            raise APIRequestError(
                f"Embedding row {row} is {type(item).__name__}, expected object; "
                f"request_id={request_id}"
            )
        idx = _embedding_index(item, row, size, request_id)
        if ordered[idx] is not None:
            raise APIRequestError(
                f"Embedding duplicated input index {idx}; request_id={request_id}"
            )
        ordered[idx] = _unit_vector(item.get("embedding"), dimension)
    if any(vec is None for vec in ordered):
        raise APIRequestError(f"Embedding response has missing input indices; request_id={request_id}")
    return ordered


class APIDense:
    """Native DashScope qwen3.7 embedding, batched documents and query prompt."""

    def __init__(self, http: DashScopeHTTP, model: str,
                 ids: Sequence[str], texts: Sequence[str], dimension: int,
                 batch_size: int, query_instruction: str):
        if batch_size < 1 or batch_size > 20:
            raise ValueError("qwen3.7-text-embedding-flash max batch size is 20")
        if dimension not in (256, 512, 768, 1024):
            raise ValueError("qwen3.7-text-embedding-flash supports dimensions 256/512/768/1024")
        self.http, self.model, self.dimension = http, model, dimension
        self.ids, self.batch_size = list(ids), batch_size
        self.query_instruction = query_instruction
        self.doc_vectors = self.embed(list(texts), "document")
        self.query_vectors: Dict[str, Tuple[float, ...]] = {}

    def embed(self, texts: Sequence[str], text_type: str) -> List[Tuple[float, ...]]:
        if text_type not in ("query", "document"):
            raise ValueError("invalid text_type")
        vectors: List[Tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            params: Dict[str, Any] = {
                "dimension": self.dimension,
                "text_type": text_type,
                # Qwen 3.7 Flash returns a dense embedding in this experiment.
                "output_type": "dense",
            }
            if text_type == "query" and self.query_instruction:
                params["instruct"] = self.query_instruction
            payload = {"model": self.model, "input": {"texts": batch}, "parameters": params}
            response = self.http.post("embedding", payload, len(batch))
            vectors.extend(_read_embedding_response(response, len(batch), self.dimension))
        return vectors

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if query not in self.query_vectors:
            self.query_vectors[query] = self.embed([query], "query")[0]
        v = self.query_vectors[query]
        scores = ((doc_id, sum(a * b for a, b in zip(doc_vector, v)))
                  for doc_id, doc_vector in zip(self.ids, self.doc_vectors))
        return sorted(scores, key=lambda p: (-p[1], p[0]))[:top_k]


class APIReranker:
    """Native DashScope text reranker; scores are mapped back by result.index."""

    def __init__(self, http: DashScopeHTTP, model: str, instruct: str):
        self.http, self.model, self.instruct = http, model, instruct

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        params: Dict[str, Any] = {"top_n": len(texts)}
        if self.instruct:
            params["instruct"] = self.instruct
        payload = {"model": self.model, "input": {"query": query, "documents": list(texts)},
                   "parameters": params}
        response = self.http.post("rerank", payload, len(texts))
        output = response.get("output")
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list) or len(results) != len(texts):
            raise APIRequestError("Rerank output.results count differs from rerank pool; check top_n / API response")
        ordered: List[Optional[float]] = [None] * len(texts)
        for result in results:
            if not isinstance(result, dict) or type(result.get("index")) is not int:
                raise APIRequestError("Rerank response missing document index")
            ix = result["index"]
            if not 0 <= ix < len(texts) or ordered[ix] is not None:
                raise APIRequestError("Rerank response duplicated/out-of-range document index")
            score = float(result.get("relevance_score"))
            if not math.isfinite(score):
                raise APIRequestError("Rerank returned invalid relevance_score")
            ordered[ix] = score
        if any(x is None for x in ordered):
            raise APIRequestError("Rerank response has missing scores")
        return ordered


def chunks_to_docs(items: Sequence[Tuple[str, float]],
                   chunk_to_doc: Mapping[str, str]) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for chunk_id, score in items:
        doc_id = chunk_to_doc[chunk_id]
        if doc_id not in best or score > best[doc_id]:
            best[doc_id] = score
    return sorted(best.items(), key=lambda p: (-p[1], p[0]))


def rrf_fusion(ranked: Mapping[str, Sequence[Tuple[str, float]]],
               k: int = 60, weights: Optional[Mapping[str, float]] = None):
    if k <= 0:
        raise ValueError("RRF k must be positive")
    weights = weights or {}
    fused = defaultdict(float)
    for source, hits in ranked.items():
        for rank, (doc_id, _) in enumerate(hits, 1):
            fused[doc_id] += weights.get(source, 1.0) / (k + rank)
    return sorted(fused.items(), key=lambda p: (-p[1], p[0]))


def weighted_fusion(ranked: Mapping[str, Sequence[Tuple[str, float]]],
                    weights: Optional[Mapping[str, float]] = None):
    weights = weights or {}
    fused = defaultdict(float)
    for source, hits in ranked.items():
        by_doc: Dict[str, float] = {}
        for doc_id, score in hits:
            by_doc.setdefault(doc_id, float(score))
        if not by_doc:
            continue
        lo, hi = min(by_doc.values()), max(by_doc.values())
        scale = hi - lo
        for doc_id, score in by_doc.items():
            val = ((score - lo) / scale) if scale > 0 else 1.0
            fused[doc_id] += weights.get(source, 1.0) * val
    return sorted(fused.items(), key=lambda p: (-p[1], p[0]))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(set(gold)) if gold else 0.0


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & set(gold)))


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    wanted = set(gold)
    for idx, doc in enumerate(ranked, 1):
        if doc in wanted:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    wanted = set(gold)
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in
              enumerate(ranked[:k], 1) if doc in wanted)
    ideal = sum(1 / math.log2(rank + 1) for rank in
                range(1, min(k, len(wanted)) + 1))
    return dcg / ideal if ideal else 0.0


def metric_values(ranked: Sequence[str], gold: Sequence[str], k: int) -> Dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(ranked, gold, k),
        f"hit@{k}": hit_at_k(ranked, gold, k),
        "mrr": reciprocal_rank(ranked, gold),
        f"ndcg@{k}": ndcg_at_k(ranked, gold, k),
    }


class Pipeline:
    def __init__(self, corpus: Sequence[Mapping[str, str]], args, http: Optional[DashScopeHTTP] = None):
        self.args = args
        self.http = http
        self.docs = {row["doc_id"]: row["text"] for row in corpus}
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_to_doc: Dict[str, str] = {}
        for row in corpus:
            splits = chunk_text(row["text"], args.chunk_size, args.chunk_overlap)
            for ix, span in enumerate(splits):
                cid = f"{row['doc_id']}::c{ix}" if len(splits) > 1 else row["doc_id"]
                self.chunk_ids.append(cid)
                self.chunk_texts.append(span)
                self.chunk_to_doc[cid] = row["doc_id"]
        self.n_docs = len(self.docs)
        self.n_chunks = len(self.chunk_ids)

        started = now_ms()
        self.sparse = BM25(self.chunk_ids, self.chunk_texts)
        self.init_ms: Dict[str, float] = {"bm25_build": now_ms() - started}
        self.dense = None
        self.reranker = None

        if not args.no_dense:
            if http is None:
                raise ValueError("Dense API mode requires authenticated DashScopeHTTP transport")
            started = now_ms()
            self.dense = APIDense(
                http, args.embed_model, self.chunk_ids, self.chunk_texts,
                args.embedding_dimension, args.batch_size, args.query_instruction,
            )
            self.init_ms["document_embedding_api"] = now_ms() - started
            if not args.no_rerank:
                self.reranker = APIReranker(http, args.reranker_model, args.rerank_instruction)
                self.init_ms["rerank_setup"] = 0.0

    def run_query(self, query: str) -> Dict[str, Any]:
        args = self.args
        component_ms: Dict[str, float] = {}
        output: Dict[str, List[Tuple[str, float]]] = {}
        tstart = now_ms()
        before_api = len(self.http.summary()["receipts"]) if self.http else 0

        def sparse_job():
            begin = now_ms()
            chunks = self.sparse.search(query, args.top_k)
            return chunks_to_docs(chunks, self.chunk_to_doc), now_ms() - begin

        def dense_job():
            begin = now_ms()
            chunks = self.dense.search(query, args.top_k)
            return chunks_to_docs(chunks, self.chunk_to_doc), now_ms() - begin

        if self.dense is not None and args.parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_s = pool.submit(sparse_job)
                future_d = pool.submit(dense_job)
                sparse_docs, sparse_ms = future_s.result()
                dense_docs, dense_ms = future_d.result()
        else:
            sparse_docs, sparse_ms = sparse_job()
            if self.dense is not None:
                dense_docs, dense_ms = dense_job()

        retrieval_wall_ms = now_ms() - tstart
        output["sparse"] = sparse_docs
        component_ms["sparse"] = sparse_ms
        elapsed_by_method = {"sparse": sparse_ms}
        changes: List[Dict[str, Any]] = []
        candidate_overlap = None
        candidate_union = len({doc for doc, _ in sparse_docs})
        rrf_pool_ids: List[str] = []

        if self.dense is not None:
            output["dense"] = dense_docs
            component_ms["dense"] = dense_ms
            elapsed_by_method["dense"] = dense_ms
            ranks = {"sparse": sparse_docs, "dense": dense_docs}
            weights = {"sparse": args.sparse_weight, "dense": args.dense_weight}
            ids_sparse = {d for d, _ in sparse_docs}
            ids_dense = {d for d, _ in dense_docs}
            candidate_overlap = len(ids_sparse & ids_dense)
            candidate_union = len(ids_sparse | ids_dense)

            start = now_ms()
            output["rrf"] = rrf_fusion(ranks, k=args.k_rrf, weights=weights)
            component_ms["rrf_fusion"] = now_ms() - start
            elapsed_by_method["rrf"] = retrieval_wall_ms + component_ms["rrf_fusion"]

            start = now_ms()
            output["weighted"] = weighted_fusion(ranks, weights=weights)
            component_ms["weighted_fusion"] = now_ms() - start
            elapsed_by_method["weighted"] = retrieval_wall_ms + component_ms["weighted_fusion"]

            if self.reranker is not None:
                # Crucial: only RRF top-N may be reranked; no outside documents.
                pool = output["rrf"][:args.rerank_pool]
                rrf_pool_ids = [doc_id for doc_id, _ in pool]
                start = now_ms()
                scores = self.reranker.score(query, [self.docs[i] for i in rrf_pool_ids])
                output["rerank"] = sorted(zip(rrf_pool_ids, scores),
                                          key=lambda p: (-p[1], p[0]))[:args.rerank_top_k]
                component_ms["rerank"] = now_ms() - start
                elapsed_by_method["rerank"] = now_ms() - tstart
                before = {d: rank for rank, (d, _) in enumerate(output["rrf"], 1)}
                after = {d: rank for rank, (d, _) in enumerate(output["rerank"], 1)}
                for did in rrf_pool_ids:
                    changes.append({
                        "doc_id": did,
                        "rrf_rank": before[did],
                        "rerank_rank": after.get(did),
                        "rank_delta": before[did] - after[did] if did in after else None,
                    })

        return {
            "methods": output,
            "trace": {
                "component_latency_ms": component_ms,
                "estimated_method_latency_ms": elapsed_by_method,
                "total_pipeline_latency_ms": now_ms() - tstart,
                "retrieval_wall_ms": retrieval_wall_ms,
                "candidate_overlap": candidate_overlap,
                "candidate_union": candidate_union,
                "rrf_rerank_pool_ids": rrf_pool_ids,
                "rank_changes": changes,
                "api_receipts_this_query": (
                    self.http.summary()["receipts"][before_api:] if self.http else []
                ),
            },
        }


def _valid_inputs(corpus: Any, queries: Any, require_queries: bool) -> None:
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a nonempty JSON list")
    ids = set()
    for row in corpus:
        if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str) or not isinstance(row.get("text"), str):
            raise ValueError("Each document must have string doc_id and text fields")
        if not row["doc_id"] or not row["text"].strip() or row["doc_id"] in ids:
            raise ValueError("Documents must have unique nonempty ids and text")
        ids.add(row["doc_id"])
    if not require_queries:
        return
    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries must be a nonempty JSON list")
    for row in queries:
        if not isinstance(row, dict) or not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("Every evaluation query needs a nonempty query string")
        gold = row.get("expected")
        if not isinstance(gold, list) or not gold or not all(isinstance(x, str) for x in gold):
            raise ValueError("Every evaluation query needs a nonempty expected list")
        missing = set(gold) - ids
        if missing:
            raise ValueError(f"Gold IDs not found in corpus for {row['query']!r}: {sorted(missing)}")


def _percentile_ms(samples: Sequence[float], percentile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    ix = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[ix]


def aggregate(rows: Sequence[Dict[str, Any]], eval_k: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for method, _ in METHODS:
        current = [r["methods"][method] for r in rows if method in r["methods"]]
        if not current:
            continue
        metric_names = [f"recall@{eval_k}", f"hit@{eval_k}", "mrr", f"ndcg@{eval_k}"]
        latency = [r["trace"]["estimated_method_latency_ms"][method]
                   for r in rows if method in r["methods"]]
        summary[method] = {
            "n_queries": len(current),
            **{m: round(statistics.mean(v["metrics"][m] for v in current), 6)
               for m in metric_names},
            "latency_ms": {
                "mean": round(statistics.mean(latency), 3),
                "p50": round(statistics.median(latency), 3),
                "p95": round(_percentile_ms(latency, 0.95), 3),
            },
        }
    return summary


def _results_to_json(result: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
    return [{"doc_id": doc, "score": round(score, 8), "rank": idx}
            for idx, (doc, score) in enumerate(result, 1)]


def evaluation(pipeline: Pipeline, queries: Sequence[Mapping[str, Any]], eval_k: int):
    per_query = []
    started = now_ms()
    for idx, spec in enumerate(queries, 1):
        query = spec["query"]
        gold = spec["expected"]
        try:
            run = pipeline.run_query(query)
        except APIRequestError as exc:
            exc.partial_rows = list(per_query)
            raise
        scored = {}
        for method, results in run["methods"].items():
            ids = [did for did, _ in results]
            scored[method] = {
                "metrics": metric_values(ids, gold, eval_k),
                "ranked_docs": _results_to_json(results),
                "first_gold_rank": next((rank for rank, did in enumerate(ids, 1)
                                         if did in gold), None),
                "top_doc": ids[0] if ids else None,
            }
        per_query.append({
            "query_index": idx,
            "query": query,
            "category": spec.get("category", "uncategorized"),
            "expected": gold,
            "methods": scored,
            "trace": run["trace"],
        })
        hits = {m: int(scores["metrics"][f"hit@{eval_k}"])
                for m, scores in scored.items()}
        print(f"[{idx:02d}/{len(queries):02d}] {query[:54]} | hit@{eval_k}: {hits}", flush=True)
    categories = sorted({r["category"] for r in per_query})
    by_category = {
        cat: aggregate([row for row in per_query if row["category"] == cat], eval_k)
        for cat in categories
    }
    return {
        "summary": aggregate(per_query, eval_k),
        "by_category": by_category,
        "per_query": per_query,
        "eval_k": eval_k,
        "evaluation_elapsed_ms": round(now_ms() - started, 2),
    }


def print_summary(result: Mapping[str, Any], pipeline: Pipeline) -> None:
    k = result["eval_k"]
    print("\n" + "=" * 94)
    print("Experiment 3-6 | BM25 + Dense API + RRF/Weighted + Rerank API")
    print("=" * 94)
    print(f"Documents: {pipeline.n_docs} | Chunks: {pipeline.n_chunks} | Queries: {len(result['per_query'])}")
    print(f"Eval: true recall@{k}, hit@{k}, MRR, nDCG@{k}; MRR is full returned rank list")
    print("-" * 94)
    print(f"{'Method':<25}{'Recall':>11}{'Hit':>11}{'MRR':>11}{'nDCG':>11}{'mean ms':>13}")
    for method, label in METHODS:
        if method not in result["summary"]:
            continue
        vals = result["summary"][method]
        print(f"{label:<25}{vals[f'recall@{k}']:>11.4f}{vals[f'hit@{k}']:>11.4f}"
              f"{vals['mrr']:>11.4f}{vals[f'ndcg@{k}']:>11.4f}"
              f"{vals['latency_ms']['mean']:>13.1f}")
    print("-" * 94)
    print("NOTE: rerank only reorders RRF top-N; it cannot rescue documents never retrieved.")
    print("NOTE: per-query latency excludes one-time document embedding API indexing (reported in JSON).")
    print("=" * 94)


def print_per_query(result: Mapping[str, Any]) -> None:
    print("\nPer-query first relevant rank ( '-' means not retrieved):")
    concise = {"sparse": "BM25", "dense": "Dense", "rrf": "RRF",
               "weighted": "Weighted", "rerank": "Rerank"}
    print(f"{'Query':<52}" + "".join(f"{concise[method]:>11}" for method, _ in METHODS))
    for row in result["per_query"]:
        short = row["query"].replace("\n", " ")[:50]
        cells = []
        for method, _ in METHODS:
            value = row["methods"].get(method, {}).get("first_gold_rank")
            cells.append(str(value) if value is not None else "-")
        print(f"{short:<52}" + "".join(f"{v:>11}" for v in cells))


def print_trace(query: str, pipeline: Pipeline, result: Mapping[str, Any]) -> None:
    print("\n" + "=" * 92)
    print(f"Query: {query!r}; {pipeline.n_docs} documents / {pipeline.n_chunks} chunks")
    for method, label in METHODS:
        hits = result["methods"].get(method)
        if hits is None:
            continue
        print(f"\n[{label}]")
        for idx, (doc_id, score) in enumerate(hits[:10], 1):
            snippet = pipeline.docs[doc_id].replace("\n", " ")[:82]
            print(f"  {idx:2}. {doc_id:<28} {score:>11.6f}  {snippet}")
    if result["trace"]["rank_changes"]:
        print("\n[Reranker rank change within RRF top-N]")
        for change in result["trace"]["rank_changes"]:
            print("  {doc_id}: {rrf_rank} -> {rerank_rank} (delta {rank_delta})".format(**change))
    print("=" * 92)


def to_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def persist(output_dir: Path, args, corpus, queries, result: Mapping[str, Any],
            pipeline: Pipeline, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    k = args.eval_k
    config = {
        "version": VERSION, "mode": mode, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": UPSTREAM,
        "upstream_evaluate_blob_sha": UPSTREAM_EVALUATE_BLOB_SHA,
        "upstream_fusion_blob_sha": UPSTREAM_FUSION_BLOB_SHA,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "corpus_sha256": sha256_json(corpus), "queries_sha256": sha256_json(queries),
        "corpus_size": len(corpus), "chunks": pipeline.n_chunks,
        "embedding_model": args.embed_model if pipeline.dense is not None else None,
        "reranker_model": args.reranker_model if pipeline.reranker is not None else None,
        "backend": "DashScope native HTTP API" if pipeline.dense is not None else "local pure-Python BM25",
        "api_host": (urlparse(pipeline.http.base_url).hostname if pipeline.http else None),
        "api_receipts": pipeline.http.summary() if pipeline.http else {"http_requests_attempted": 0, "successful_calls": 0, "receipts": []},
        "dependencies": {"PyYAML": _package_version("PyYAML"), "jieba": _package_version("jieba")},
        "parameters": {
            "top_k": args.top_k, "eval_k": args.eval_k,
            "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
            "k_rrf": args.k_rrf, "sparse_weight": args.sparse_weight,
            "dense_weight": args.dense_weight, "rerank_pool": args.rerank_pool,
            "rerank_top_k": args.rerank_top_k, "embedding_batch_size": args.batch_size,
            "embedding_dimension": args.embedding_dimension,
            "parallel": args.parallel,
            "query_instruction": args.query_instruction,
            "rerank_instruction": args.rerank_instruction,
        },
        "initialization_ms": pipeline.init_ms,
    }
    payload = {"config": config, "corpus": corpus, "queries": queries, **result}
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if mode == "single_query":
        return
    sr = []
    for method, label in METHODS:
        m = result["summary"].get(method)
        if m:
            sr.append({"method": label, "n_queries": m["n_queries"],
                       "recall": m[f"recall@{k}"], "hit": m[f"hit@{k}"],
                       "mrr": m["mrr"], "ndcg": m[f"ndcg@{k}"],
                       "mean_latency_ms": m["latency_ms"]["mean"],
                       "p95_latency_ms": m["latency_ms"]["p95"]})
    to_csv(output_dir / "summary.csv", ["method", "n_queries", "recall", "hit", "mrr", "ndcg",
                                         "mean_latency_ms", "p95_latency_ms"], sr)
    category_rows = []
    for category, by_method in result.get("by_category", {}).items():
        for method, label in METHODS:
            m = by_method.get(method)
            if m:
                category_rows.append({
                    "category": category, "method": label, "n_queries": m["n_queries"],
                    "recall": m[f"recall@{k}"], "hit": m[f"hit@{k}"],
                    "mrr": m["mrr"], "ndcg": m[f"ndcg@{k}"],
                })
    to_csv(output_dir / "category_summary.csv",
           ["category", "method", "n_queries", "recall", "hit", "mrr", "ndcg"],
           category_rows)
    detail = []
    for row in result["per_query"]:
        for method, label in METHODS:
            value = row["methods"].get(method)
            if value is None:
                continue
            detail.append({"query": row["query"], "category": row["category"],
                           "expected": "|".join(row["expected"]), "method": label,
                           "first_gold_rank": value["first_gold_rank"],
                           "top_doc": value["top_doc"],
                           **{key: round(val, 6) for key, val in value["metrics"].items()}})
    to_csv(output_dir / "per_query.csv", ["query", "category", "expected", "method",
                                             "first_gold_rank", "top_doc", f"recall@{k}",
                                             f"hit@{k}", "mrr", f"ndcg@{k}"], detail)
    md = ["# Experiment 3-6: stage-by-stage retrieval evaluation", "",
          f"Documents: {pipeline.n_docs}; chunks: {pipeline.n_chunks}; queries: {len(queries)}", "",
          f"| Stage | Recall@{k} | Hit@{k} | MRR | nDCG@{k} | Mean latency (ms) |",
          "|---|---:|---:|---:|---:|---:|"]
    for method, label in METHODS:
        m = result["summary"].get(method)
        if m:
            md.append(f"| {label} | {m[f'recall@{k}']:.4f} | {m[f'hit@{k}']:.4f} | "
                      f"{m['mrr']:.4f} | {m[f'ndcg@{k}']:.4f} | {m['latency_ms']['mean']:.1f} |")
    md.extend(["", "Interpretation notes:",
               "- Recall@k is standard multi-gold recall; Hit@k is the upstream book's simplified success measure.",
               "- A reranker does not retrieve unseen documents; the RRF candidate pool bounds its performance.",
               "- This is a small educational 20-document / 12-query benchmark; avoid broad generalizations.",
               "- This single-process implementation does not test microservice API availability.",
               "- Document embedding API calls are one-time indexing cost, separate from per-query latency.",
               "- API receipts and total token usage are recorded without secrets.",
               "- Full traces, component timing, and gold references are in results.json.", ""])
    (output_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def self_test() -> None:
    """No external API calls. Check payload protocol, response indexing, budgets."""
    from unittest.mock import patch
    assert len(CORPUS) == 20 and len(QUERIES) == 12
    _valid_inputs(CORPUS, QUERIES, True)
    assert tokenize("XR-7003 HTTP-403") == ["xr-7003", "http-403"]
    assert chunk_text("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]
    b = BM25(["x1", "x2", "x3"], ["XR-7003", "XR-7005", "HTTP-500"])
    assert b.search("XR-7003", 2)[0][0] == "x1"
    r = rrf_fusion({"s": [("a", 2), ("b", 1)], "d": [("b", 1), ("c", 0.2)]}, 60)
    assert r[0][0] == "b"
    w = weighted_fusion({"s": [("a", 10), ("b", 0)], "d": [("b", 3), ("c", 1)]})
    assert len(w) == 3
    assert recall_at_k(["a", "b"], ["b", "c"], 2) == 0.5
    assert hit_at_k(["a", "b"], ["b", "c"], 2) == 1
    assert reciprocal_rank(["x", "a"], ["a"]) == .5
    assert abs(ndcg_at_k(["a", "b"], ["a", "b"], 2)-1) < 1e-10
    assert chunks_to_docs([("x::0", 1), ("x::1", 2), ("y", 1)],
                          {"x::0": "x", "x::1": "x", "y": "y"}) == [("x", 2), ("y", 1)]
    class Settings:
        chunk_size = 1000
        chunk_overlap = 30
        no_dense = False
        no_rerank = False
        top_k = 3
        k_rrf = 60
        sparse_weight = 1.0
        dense_weight = 1.0
        parallel = True
        rerank_pool = 2
        rerank_top_k = 2
        embedding_dimension = 256
        batch_size = 2
        query_instruction = "Find relevant passage"
        rerank_instruction = "Rank passage relevance"
        embed_model = "qwen3.7-text-embedding-flash"
        reranker_model = "qwen3.7-text-rerank"
    records = []
    class Response:
        status = 200
        def __init__(self, value):
            self.raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        def read(self):
            return self.raw
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def urlopen_stub(req, timeout):
        assert timeout == 30
        assert req.headers.get("Authorization") == "Bearer fake-key-for-unit-test"
        body = json.loads(req.data)
        records.append((req.full_url, body))
        if req.full_url.endswith(EMBED_PATH):
            assert body["model"] == "qwen3.7-text-embedding-flash"
            assert body["parameters"]["dimension"] == 256
            assert body["parameters"]["text_type"] in ("query", "document")
            assert len(body["input"]["texts"]) <= 2
            if body["parameters"]["text_type"] == "query":
                assert body["parameters"]["instruct"] == "Find relevant passage"
            else:
                assert "instruct" not in body["parameters"]
            items = []
            for ix, text in enumerate(body["input"]["texts"]):
                items.append({"text_index": ix, "embedding":
                    ([1.0, 0.0] if "XR-7003" in text else [0.0, 1.0]) + [0.0] * 254})
            # Official response gives explicit text_index, not guaranteed index order.
            return Response({"output": {"embeddings": list(reversed(items))},
                             "usage": {"total_tokens": 7},
                             "request_id": "embed-unit-test"})
        assert req.full_url.endswith(RERANK_PATH)
        assert body["model"] == "qwen3.7-text-rerank"
        assert body["parameters"]["top_n"] == len(body["input"]["documents"])
        assert body["parameters"]["instruct"] == "Rank passage relevance"
        items = [{"index": ix, "relevance_score":
                  (0.9 if "XR-7003" in value else 0.1)}
                 for ix, value in enumerate(body["input"]["documents"])]
        return Response({"output": {"results": list(reversed(items))},
                         "usage": {"total_tokens": 13}, "request_id": "rerank-unit-test"})
    with patch.object(urlrequest, "urlopen", urlopen_stub):
        http = DashScopeHTTP("https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
                            "fake-key-for-unit-test", timeout_sec=30,
                            max_retries=0, max_http_requests=15)
        docs = [
          {"doc_id": "a", "text": "Device XR-7003 documentation"},
          {"doc_id": "b", "text": "Device XR-7005 documentation"},
          {"doc_id": "c", "text": "HTTP-403 access denied"},
        ]
        pipe = Pipeline(docs, Settings(), http)
        got = pipe.run_query("XR-7003")
        assert set(got["methods"]) == {"sparse", "dense", "rrf", "weighted", "rerank"}
        assert got["methods"]["dense"][0][0] == "a"
        assert got["methods"]["rerank"][0][0] == "a"
        assert all(d in got["trace"]["rrf_rerank_pool_ids"] for d,_ in got["methods"]["rerank"])
        # Query is cached: same query twice should not cause a second embedding call.
        first_calls = http.summary()["successful_calls"]
        pipe.run_query("XR-7003")
        assert http.summary()["successful_calls"] == first_calls + 1  # rerank only
        qresult = evaluation(pipe, [{"query": "XR-7003", "expected": ["a"], "category": "technical-code"}], 2)
        assert len(qresult["summary"]) == 5
        assert http.summary()["calls_by_stage"] == {"embedding": 3, "rerank": 3}
        assert http.summary()["total_tokens_reported"] == 3*7 + 3*13
        assert "fake-key-for-unit-test" not in json.dumps(http.summary())
    # Reject invalid or incomplete API results; never silently fabricate scores.
    assert _safe_error_message("sk-topsecret12345") == "sk-[REDACTED]"
    try:
        _unit_vector([1.0, 2.0], 256)
        raise AssertionError("Dimension validation did not fire")
    except APIRequestError:
        pass
    # Response compatibility: native text_index, gateway index, singleton without
    # an index, and OpenAI-style data are accepted with exact vector mapping.
    v0 = [1.0] + [0.0] * 255
    v1 = [0.0, 1.0] + [0.0] * 254
    native_reply = {"output": {"embeddings": [
        {"text_index": 1, "embedding": v1},
        {"text_index": 0, "embedding": v0},
    ]}}
    assert _read_embedding_response(native_reply, 2, 256)[0][0] == 1.0
    gateway_reply = {"output": {"embeddings": [
        {"index": "1", "embedding": v1},
        {"index": "0", "embedding": v0},
    ]}}
    assert _read_embedding_response(gateway_reply, 2, 256)[1][1] == 1.0
    compatible_reply = {"data": [
        {"index": 1, "embedding": v1}, {"index": 0, "embedding": v0},
    ]}
    assert _read_embedding_response(compatible_reply, 2, 256)[0][0] == 1.0
    unindexed_singleton = {"output": {"embeddings": [{"embedding": v0}]}}
    assert _read_embedding_response(unindexed_singleton, 1, 256)[0][0] == 1.0
    # Multi-vector payloads without indices must be rejected, never zip-aligned.
    try:
        _read_embedding_response({"output": {"embeddings": [
            {"embedding": v0}, {"embedding": v1},
        ]}, "request_id": "sample-id"}, 2, 256)
        raise AssertionError("Unindexed batch was incorrectly accepted")
    except APIRequestError as exc:
        assert "--batch-size 1" in str(exc)
        assert "sample-id" in str(exc)
        assert "embedding" in str(exc)
    for reply in (
        {"output": {"embeddings": [{"text_index": 0, "embedding": v0},
                                   {"index": 0, "embedding": v1}]}},
        {"output": {"embeddings": [{"text_index": 0, "index": 1, "embedding": v0},
                                   {"index": 1, "embedding": v1}]}},
        {"output": {"embeddings": [{"index": 0, "embedding": v0}]}},
    ):
        try:
            _read_embedding_response(reply, 2, 256)
            raise AssertionError("Invalid embedding index schema was accepted")
        except APIRequestError:
            pass
    class BadHttp:
        def post(self, *_):
            return {"output": {"results": [{"index": 0, "relevance_score": 0.5}]}}
    try:
        APIReranker(BadHttp(), "qwen3.7-text-rerank", "").score("test", ["a", "b"])
        raise AssertionError("Incomplete rerank scores accepted")
    except APIRequestError:
        pass
    # Transport retries only transient failures and counts network attempts.
    from io import BytesIO
    retry_hits = {"value": 0}
    def rate_limit_once(req, timeout):
        retry_hits["value"] += 1
        if retry_hits["value"] == 1:
            raise urlerror.HTTPError(req.full_url, 429, "Too many requests", None,
                                     BytesIO(b'{"code":"Throttling","message":"rate limited"}'))
        return Response({"output": {"results": [{"index": 0, "relevance_score": 0.6}]},
                         "usage": {"total_tokens": 5}, "request_id": "retry-pass"})
    with patch.object(urlrequest, "urlopen", rate_limit_once):
        retry_http = DashScopeHTTP("https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
                                   "fake-key-for-unit-test", timeout_sec=30,
                                   retry_base_sec=0, max_retries=1, max_http_requests=3)
        retry_http.post("rerank", {"model": "qwen3.7-text-rerank", "input": {
            "query": "a", "documents": ["b"]}}, 1)
        assert retry_http.summary()["http_requests_attempted"] == 2
        assert retry_http.summary()["successful_calls"] == 1
    def unauth(req, timeout):
        raise urlerror.HTTPError(req.full_url, 401, "Unauthorized", None,
                                 BytesIO(b'{"code":"InvalidApiKey","message":"unauthorized"}'))
    with patch.object(urlrequest, "urlopen", unauth):
        unauthorized_http = DashScopeHTTP("https://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
                                          "fake-key-for-unit-test", timeout_sec=30,
                                          max_retries=2)
        try:
            unauthorized_http.post("embedding", {"model": "qwen3.7-text-embedding-flash"}, 1)
            raise AssertionError("401 incorrectly accepted")
        except APIRequestError as exc:
            assert "HTTP 401" in str(exc)
        assert unauthorized_http.summary()["http_requests_attempted"] == 1
    # Test the most common existing .env configurations without network access.
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    names = ["DASHSCOPE_API_KEY", "DASHSCOPE_HTTP_BASE_URL", "DASHSCOPE_BASE_URL",
             "QWEN_API_BASE_URL", "DASHSCOPE_WORKSPACE_ID", "DASHSCOPE_REGION",
             "DASHSCOPE_EMBEDDING_URL", "DASHSCOPE_RERANK_URL", "OPENAI_BASE_URL"]
    def isolated_env(mapping):
        return patch.dict(os.environ, mapping, clear=True)
    cfg = {"base_url_env": "DASHSCOPE_HTTP_BASE_URL",
           "embedding_url_env": "DASHSCOPE_EMBEDDING_URL",
           "rerank_url_env": "DASHSCOPE_RERANK_URL", "region": "cn-beijing"}
    cli = SimpleNamespace(api_base_url=None, workspace_id=None, region=None)
    with TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        with isolated_env({}):
            dot = temp_dir / ".env"
            dot.write_text('export DASHSCOPE_API_KEY="unit-test-secret"\n'
                           'QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" # old chapter 2\n',
                           encoding="utf-8")
            details = _load_dotenv([dot])
            assert len(details["files"]) == 1 and os.getenv("DASHSCOPE_API_KEY") == "unit-test-secret"
            e = resolve_api_endpoints(cfg, cli)
            assert e["base_url"] == "https://dashscope.aliyuncs.com/api/v1"
            assert e["base_source"] == "QWEN_API_BASE_URL"
        with isolated_env({}):
            dot = temp_dir / ".env"
            dot.write_text('DASHSCOPE_API_KEY=unit-test-secret\n'
                           'DASHSCOPE_HTTP_BASE_URL=https://unit.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n',
                           encoding="utf-8")
            _load_dotenv([dot])
            assert resolve_api_endpoints(cfg, cli)["base_url"] == "https://unit.cn-beijing.maas.aliyuncs.com/api/v1"
        with isolated_env({}):
            dot = temp_dir / ".env"
            dot.write_text('DASHSCOPE_EMBEDDING_URL=https://unit.cn-beijing.maas.aliyuncs.com/api/v1' + EMBED_PATH + '\n'
                           'DASHSCOPE_RERANK_URL=https://unit.cn-beijing.maas.aliyuncs.com/api/v1' + RERANK_PATH + '\n',
                           encoding="utf-8")
            _load_dotenv([dot])
            e = resolve_api_endpoints(cfg, cli)
            assert e["base_url"] == "https://unit.cn-beijing.maas.aliyuncs.com/api/v1"
            assert e["embedding_url"].endswith(EMBED_PATH)
            assert e["rerank_url"].endswith(RERANK_PATH)
        with isolated_env({"DASHSCOPE_HTTP_BASE_URL": ""}):
            dot = temp_dir / ".env"
            dot.write_text('DASHSCOPE_HTTP_BASE_URL=https://dashscope.aliyuncs.com/api/v1\n', encoding="utf-8")
            _load_dotenv([dot])
            assert os.getenv("DASHSCOPE_HTTP_BASE_URL") == "https://dashscope.aliyuncs.com/api/v1"
        with isolated_env({"DASHSCOPE_HTTP_BASE_URL": "https://old.cn-beijing.maas.aliyuncs.com/api/v1"}):
            dot = temp_dir / ".env"
            dot.write_text('DASHSCOPE_HTTP_BASE_URL=https://new.cn-beijing.maas.aliyuncs.com/api/v1\n', encoding="utf-8")
            _load_dotenv([dot])
            assert "old" in os.environ["DASHSCOPE_HTTP_BASE_URL"]
            _load_dotenv([dot], override=True)
            assert "new" in os.environ["DASHSCOPE_HTTP_BASE_URL"]
        with isolated_env({"DASHSCOPE_API_KEY": "unit-test-secret"}):
            e = resolve_api_endpoints(cfg, cli)
            assert e["base_source"].startswith("official shared host")
        with isolated_env({"OPENAI_BASE_URL": "https://api.openai.com/v1"}):
            assert resolve_api_endpoints(cfg, cli)["base_source"].startswith("official shared host")
        with isolated_env({"DASHSCOPE_WORKSPACE_ID": "workspace123"}):
            assert resolve_api_endpoints(cfg, cli)["base_url"] == "https://workspace123.cn-beijing.maas.aliyuncs.com/api/v1"
    print("PASS: .env project loading, QWEN/DASHSCOPE aliases, URL path conversion, stage URLs, precedence and fallback")
    print("PASS: 20/12 official fixtures; BM25, RRF, weighted, metrics; native HTTP embedding/rerank protocols")
    print("PASS: batching, native/gateway/singleton embedding index mapping, model names, candidate restriction, caching, receipts, errors")
    print("NOTE: --self-test uses local simulated HTTP replies only; full experiment requires real DashScope API")


def _project_config_path(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--config not found: {path}")
        return path
    candidates = []
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend(p / "config.yaml" for p in (base, *list(base.parents)[:5]))
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("No config.yaml found. Supply --config /path/to/config.yaml")


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing PyYAML: python -m pip install pyyaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml root must be a mapping")
    # Shared settings live at the YAML root, so later RAG experiments can reuse
    # the same API / embedding / reranker / retrieval configuration without
    # introducing per-experiment wrapper sections.
    legacy = data.get("experiment_3_6")
    if isinstance(legacy, dict) and not any(
        key in data for key in ("api", "embedding", "reranker", "retrieval")
    ):
        raise ValueError(
            "Legacy experiment_3_6 configuration detected. "
            "Move its api, embedding, reranker, retrieval and dataset "
            "sections to the top level of config.yaml."
        )
    required = ("api", "embedding", "reranker", "retrieval")
    absent = [section for section in required if section not in data]
    if absent:
        raise ValueError(
            "Missing shared top-level config sections: " + ", ".join(absent)
            + ". Use the updated config.yaml; do not wrap shared model settings "
              "in an experiment-specific section."
        )
    for section in (*required, "dataset"):
        value = data.get(section)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"config.yaml.{section} must be a mapping")
    return data


def _get_nested(data: Mapping[str, Any], a: str, b: str, default: Any) -> Any:
    sect = data.get(a) or {}
    if not isinstance(sect, dict):
        raise ValueError(f"config.yaml.{a} must be a mapping")
    return sect.get(b, default)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Official 3-6 five-stage retrieval ablation; API embedding + API reranker; standalone single Python file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="Path to YAML (defaults to nearest config.yaml)")
    p.add_argument("--env-file", help="Explicit .env path, relative to config.yaml; no dotenv package required")
    p.add_argument("--env-override", action="store_true", help="Let .env override inherited shell environment variables")
    p.add_argument("--self-test", action="store_true", help="Comprehensive offline protocol tests: no requests or billing")
    p.add_argument("--dry-run", action="store_true", help="Print resolved API configuration without sending requests")
    p.add_argument("--confirm-api-cost", action="store_true", help="Explicitly approve paid API calls")
    p.add_argument("--corpus", help="Corpus JSON file; overrides config and built-in upstream sample")
    p.add_argument("--queries", help="Gold-labeled query JSON; overrides config and sample")
    p.add_argument("--query", help="Single query trace (not a labeled evaluation)")
    p.add_argument("--limit-queries", type=int, help="Run first N gold queries (0=all)")
    p.add_argument("--no-dense", action="store_true", help="Pure-Python BM25-only test: NO paid API")
    p.add_argument("--no-rerank", action="store_true", help="Disable rerank API, keep Dense API and fusion")
    p.add_argument("--embed-model", help="Embedding API model override")
    p.add_argument("--reranker-model", help="Reranking API model override")
    p.add_argument("--embedding-dimension", type=int, help="Embedding dimension override")
    p.add_argument("--batch-size", type=int, help="Embedding API max texts per request, <=20")
    p.add_argument("--api-base-url", help="Native DashScope base URL ending in /api/v1")
    p.add_argument("--workspace-id", help="DashScope workspace ID (overrides YAML)")
    p.add_argument("--region", choices=["cn-beijing", "ap-southeast-1", "us-east-1"], help="DashScope region")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--eval-k", type=int)
    p.add_argument("--k-rrf", type=int)
    p.add_argument("--dense-weight", type=float)
    p.add_argument("--sparse-weight", type=float)
    p.add_argument("--rerank-pool", type=int)
    p.add_argument("--rerank-top-k", type=int)
    p.add_argument("--parallel", action="store_true", default=None, help="Run local BM25 and remote query embedding in parallel")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel BM25 + Dense query")
    p.add_argument("--output-dir", help="Override exact output directory; default <script_dir>/results/exp3_6_TIMESTAMP")
    p.add_argument("--no-per-query", action="store_true")
    return p


def _arg_value(command: Any, key: str, cfg: Mapping[str, Any], section: str,
               field: str, default: Any) -> Any:
    override = getattr(command, key, None)
    return override if override is not None else _get_nested(cfg, section, field, default)


def _validate_args(args: Any, parser: argparse.ArgumentParser) -> None:
    if args.chunk_size <= 0 or not (0 <= args.chunk_overlap < args.chunk_size):
        parser.error("0 <= chunk_overlap < chunk_size is required")
    for name in ("top_k", "eval_k", "k_rrf", "rerank_pool", "rerank_top_k", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.batch_size > 20:
        parser.error("qwen3.7-text-embedding-flash API accepts at most 20 texts per batch")
    if args.embedding_dimension not in (256, 512, 768, 1024):
        parser.error("qwen3.7-text-embedding-flash supports dimensions 256, 512, 768 and 1024")
    if args.rerank_pool > 500:
        parser.error("qwen3.7-text-rerank max documents per request = 500")
    if not (args.dense_weight >= 0 and args.sparse_weight >= 0 and args.dense_weight + args.sparse_weight > 0):
        parser.error("Fusion weights must be nonnegative, sum > 0")
    if args.limit_queries < 0:
        parser.error("limit_queries must be >=0")
    if args.no_dense and not args.no_rerank:
        args.no_rerank = True
        print("[说明] --no-dense implies --no-rerank; only genuine BM25 will run.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = get_parser()
    command = parser.parse_args(argv)
    if command.self_test:
        self_test()
        return 0
    path = _project_config_path(command.config)
    cfg = _load_config(path)
    api_cfg = cfg.get("api") or {}
    if not isinstance(api_cfg, dict):
        raise ValueError("config.yaml.api must be a mapping")
    env_path = command.env_file or str(api_cfg.get("env_file") or "").strip() or None
    env_details = _load_dotenv(_env_paths(path, env_path, required=bool(command.env_file)),
                               override=command.env_override)
    class Settings: pass
    args = Settings()
    for field, section, key, default in (
        ("chunk_size", "retrieval", "chunk_size", 280),
        ("chunk_overlap", "retrieval", "chunk_overlap", 40),
        ("top_k", "retrieval", "top_k", 10),
        ("eval_k", "retrieval", "eval_k", 3),
        ("k_rrf", "retrieval", "rrf_k", 60),
        ("dense_weight", "retrieval", "dense_weight", 1.0),
        ("sparse_weight", "retrieval", "sparse_weight", 1.0),
        ("rerank_pool", "retrieval", "rerank_pool", 10),
        ("rerank_top_k", "retrieval", "rerank_top_k", 10),
        ("embed_model", "embedding", "model", "qwen3.7-text-embedding-flash"),
        ("embedding_dimension", "embedding", "dimension", 1024),
        ("batch_size", "embedding", "batch_size", 20),
        ("reranker_model", "reranker", "model", "qwen3.7-text-rerank"),
    ):
        setattr(args, field, _arg_value(command, field, cfg, section, key, default))
    args.query_instruction = _get_nested(cfg, "embedding", "query_instruction",
        "Given a search query, retrieve relevant passages that answer the query.")
    args.rerank_instruction = _get_nested(cfg, "reranker", "instruction",
        "Given a web search query, retrieve relevant passages that answer the query.")
    args.parallel = bool(_get_nested(cfg, "retrieval", "parallel", True))
    if command.parallel:
        args.parallel = True
    if command.no_parallel:
        args.parallel = False
    args.no_dense = bool(command.no_dense)
    args.no_rerank = bool(command.no_rerank)
    args.no_per_query = bool(command.no_per_query)
    args.limit_queries = (command.limit_queries if command.limit_queries is not None
                          else _get_nested(cfg, "dataset", "limit_queries", 0))
    _validate_args(args, parser)
    if command.corpus:
        corpus_path = Path(command.corpus)
    else:
        src = _get_nested(cfg, "dataset", "corpus", "")
        corpus_path = path.parent / src if src else None
    if command.queries:
        queries_path = Path(command.queries)
    else:
        src = _get_nested(cfg, "dataset", "queries", "")
        queries_path = path.parent / src if src else None
    corpus = json.loads(corpus_path.read_text(encoding="utf-8-sig")) if corpus_path else CORPUS
    queries = json.loads(queries_path.read_text(encoding="utf-8-sig")) if queries_path else QUERIES
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    _valid_inputs(corpus, queries, require_queries=not bool(command.query))
    if corpus_path and not queries_path and not command.query:
        raise ValueError("Custom corpus requires --queries (or config.yaml dataset.queries) with gold IDs")
    endpoints = None
    if not args.no_dense:
        endpoints = resolve_api_endpoints(api_cfg, command)

    print("Experiment 3-6 | standalone Qwen API version", flush=True)
    print(f"Configuration: {path}")
    print(f".env loaded: {', '.join(env_details['files']) if env_details['files'] else '(none found; shell env only)'}")
    print(f"Dataset: {len(corpus)} docs, {len(queries)} labeled queries")
    print(f"Embedding: {'DISABLED' if args.no_dense else args.embed_model} (dimension={args.embedding_dimension})")
    print(f"Rerank: {'DISABLED' if args.no_rerank else args.reranker_model}")
    key_env = str(api_cfg.get("api_key_env") or "DASHSCOPE_API_KEY").strip()
    key, key_source = _env_first(key_env, "DASHSCOPE_API_KEY", "QWEN_API_KEY")
    if command.dry_run:
        if args.no_dense:
            print("API Host: (not needed for BM25-only mode)")
        else:
            print(f"API Host: {endpoints['base_url']}  [from {endpoints['base_source']}]")
            print(f"Embedding endpoint: {endpoints['embedding_url']}  [from {endpoints['embedding_source']}]")
            print(f"Rerank endpoint: {endpoints['rerank_url']}  [from {endpoints['rerank_source']}]")
        print(f"Key env var: {key_source or key_env}; set={'yes' if key else 'NO'} (secret never printed)")
        if not key and not args.no_dense:
            print("[ERROR] Missing API Key. Define DASHSCOPE_API_KEY in project root .env or shell.", file=sys.stderr)
            return 2
        print(f"Expected stages: {'BM25' if args.no_dense else ('4 stages' if args.no_rerank else '5 stages')}")
        print("Dry run: no HTTP request made and no usage billed")
        return 0
    if not args.no_dense and not command.confirm_api_cost:
        print("ERROR: Paid embedding/rerank API calls require --confirm-api-cost (or run --dry-run)", file=sys.stderr)
        return 2
    http = None
    if not args.no_dense:
        http = DashScopeHTTP(
            endpoints["base_url"], key,
            timeout_sec=float(api_cfg.get("timeout_sec", 60)),
            max_retries=int(api_cfg.get("max_retries", 3)),
            retry_base_sec=float(api_cfg.get("retry_base_sec", 1.5)),
            max_http_requests=int(api_cfg.get("max_http_requests", 100)),
            embedding_url=endpoints["embedding_url"],
            rerank_url=endpoints["rerank_url"],
        )
    if command.output_dir:
        # Explicit CLI override: relative paths are resolved from the caller's CWD.
        output_dir = Path(command.output_dir).expanduser().resolve()
    else:
        # Every standalone experiment owns its result directory. This is
        # independent of the project config location and invocation CWD.
        script_dir = Path(__file__).resolve().parent
        output_dir = (script_dir / "results" /
                      ("exp3_6_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    try:
        pipeline = Pipeline(corpus, args, http=http)
    except APIRequestError as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failed_run.json").write_text(
            json.dumps({"status": "failed_during_indexing", "error": _safe_error_message(exc),
                        "api": http.summary() if http else None,
                        "corpus_sha256": sha256_json(corpus),
                        "note": "No fallback to synthetic or fake embeddings occurred."},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    if command.query:
        record = pipeline.run_query(command.query)
        print_trace(command.query, pipeline, record)
        output = {"summary": {}, "per_query": [], "eval_k": args.eval_k,
                  "single_query": command.query,
                  "methods": {m: _results_to_json(v) for m, v in record["methods"].items()},
                  "trace": record["trace"]}
        persist(output_dir, args, corpus, [], output, pipeline, "single_query")
    else:
        try:
            result = evaluation(pipeline, queries, args.eval_k)
        except APIRequestError as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failed_run.json").write_text(
                json.dumps({"status": "failed_during_evaluation", "error": _safe_error_message(exc),
                            "completed_queries": len(getattr(exc, "partial_rows", [])),
                            "partial_rows": getattr(exc, "partial_rows", []),
                            "api": http.summary() if http else None,
                            "corpus_sha256": sha256_json(corpus),
                            "queries_sha256": sha256_json(queries),
                            "note": "No fallback to fake retrieval, partial rows are incomplete."},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        print_summary(result, pipeline)
        if not args.no_per_query:
            print_per_query(result)
        persist(output_dir, args, corpus, queries, result, pipeline, "evaluation")
    if http:
        usage = http.summary()
        print(f"API usage: successful_calls={usage['successful_calls']}, attempts={usage['http_requests_attempted']}, tokens={usage['total_tokens_reported']}")
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {_safe_error_message(exc)}", file=sys.stderr)
        raise SystemExit(2)
