"""
Training-free medical QA retrieval with a decoder-only LLM (Qwen3-VL text tower).

Final pipeline (every stage is label-free except the handful of scalars chosen by tune_post / tune_fusion
on a dev split of the queries):

    early-layer hidden states (default layers 4, 8, 12, run truncated)
      -> echo input + attention-sink token, token z-score, outlier-token masking, IDF weighting
      -> mean+max pooling                                   (pooling = "mean_max")
      -> per-layer post-processing: centre, standardise (queries with their own std), drop top PCs,
         soft whitening, re-weight the max half
      -> dense cosine  (minus a hubness penalty)
      -> fused with word-level BM25 (stopwords, plural stemming, bigrams)
      -> Rocchio pseudo-relevance feedback (optional, tuned)
      -> Sinkhorn / inverted-softmax normalisation over the query batch      <- "+sinkhorn"

TRANSDUCTIVE parts: the hub penalty (when fit queries are given), the query-side statistics and the Sinkhorn
step look at the whole query set. They help on a benchmark where queries and answers are roughly one-to-one,
and do nothing for a single query. Use `transductive=False` in run_eval.py for a corpus-only number.

FIXES APPLIED FOR LONG CONTEXT:
1. Added `mean_weight` to `EmbedderConfig` and `PostProcessor` to allow disabling mean pooling (crucial for long docs).
2. Added `LongContextEmbedder` class that implements document chunking and MaxSim aggregation for search/evaluate.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Collection, Iterator, Literal, Sequence

os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3VLTextModel

SinkMode = Literal["prepend", "drop", "none"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class EmbedderConfig:
    model_dir: str = "./saved_qwen3vl_text_embedder"
    pooling: str = "mean_max"               # the only supported pooling (kept for config compatibility)
    layers: tuple[int, ...] = (8, 18, 8)    # indices into hidden_states
    max_text_tokens: int = 2048
    batch_size: int = 4
    max_batch_tokens: int = 8192            # cap on batch_size * padded_len (echo doubles the length)
    dtype: torch.dtype = torch.bfloat16

    # --- input construction ---------------------------------------------------
    sink: SinkMode = "prepend"              # "prepend" | "drop" | "none"
    echo: bool = True                       # feed the text twice, pool the 2nd copy
    echo_sep: str = "\n"
    truncate_layers: bool = True            # stop the forward pass at max(layers)+1

    # --- token conditioning -----------------------------------------------------
    token_zscore: bool = True               # per-dim z-score of token states (needs calibrate)
    mask_outlier_tokens: bool = True        # drop massive-activation tokens (needs calibrate)
    outlier_ratio: float = 4.0              # token is an outlier if ||h|| > ratio * median ||h||
    token_norm: bool = True                 # L2-normalise each token before pooling
    use_idf: bool = True

    # --- document-level post-processing ------------------------------------------
    standardize: bool = True
    separate_query_std: bool = True         # queries are short, docs long: scale queries by their own std
    remove_pcs: int = 4                     # "all-but-the-top"
    zca_alpha: float = 0.5                  # soft whitening strength (0 = off)
    zca_rank: int = 128                     # size of the subspace that is rescaled
    half_weight: float = 1.0                # weight of the max-pool half relative to the mean-pool half
    mean_weight: float = 1.0                # weight of the mean-pool half (set to 0.0 to disable mean pooling)

    # --- lexical channel -----------------------------------------------------------
    hybrid_weight: float = 0.3              # 0 = dense only (tune_fusion picks the final value)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    lex_bigrams: bool = True                # index adjacent word pairs ("blood_pressure", "type_2")

    # --- dense refinements ---------------------------------------------------------
    prf_k: int = 0                          # Rocchio feedback docs (0 = off)
    prf_beta: float = 0.5                   # weight of the feedback centroid
    hub_k: int = 10                         # neighbourhood size for the hubness penalty
    hub_gamma: float = 0.0                  # score -= gamma * r_doc (0 = off, 0.5 = CSLS)
    norm_tau: float = 0.0                   # >0: Sinkhorn over the query batch (transductive)
    norm_iters: int = 1                     # 1 = inverted softmax, >1 = Sinkhorn iterations

    def __post_init__(self) -> None:
        if self.pooling != "mean_max":
            raise ValueError("Only pooling='mean_max' is supported in this release.")
        self.layers = tuple(self.layers)


def _flat(ids):
    if ids and isinstance(ids[0], list):
        return ids[0]
    return ids


def _resolve_sink_id(tok) -> int:
    """Qwen has no BOS; <|endoftext|> is what documents start after during pre-training."""
    unk = getattr(tok, "unk_token_id", None)
    for name in ("<|endoftext|>", "<|im_start|>"):
        try:
            i = tok.convert_tokens_to_ids(name)
        except Exception:
            i = None
        if isinstance(i, int) and i >= 0 and i != unk:
            return i
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            return int(v)
    return 0


class PromptBuilder:
    """Tokenises raw text. No instruction prefix; optional sink token and echo repetition."""

    def __init__(self, tokenizer, cfg: EmbedderConfig):
        self.tok = tokenizer
        self.cfg = cfg
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else 0
        self.sink_id = _resolve_sink_id(tokenizer)
        self.sep_ids: list[int] = []
        if cfg.echo and cfg.echo_sep:
            self.sep_ids = list(_flat(tokenizer(cfg.echo_sep, add_special_tokens=False)["input_ids"]))

    def text_ids(self, text: str) -> list[int]:
        norm = " ".join(unicodedata.normalize("NFKC", str(text)).split())
        if not norm:
            raise ValueError("Text is empty after normalisation.")
        ids = list(_flat(self.tok(norm, add_special_tokens=False)["input_ids"]))[: self.cfg.max_text_tokens]
        if not ids:
            raise ValueError(f"Text produced no tokens: {text!r}")
        return ids

    def build(self, text: str) -> tuple[list[int], tuple[int, int]]:
        """Returns (token_ids, (pool_start, pool_end))."""
        cfg = self.cfg
        ids = self.text_ids(text)
        seq: list[int] = [self.sink_id] if cfg.sink == "prepend" else []
        if cfg.echo:
            seq += ids + self.sep_ids
            start = len(seq)
            seq += ids
            return seq, (start, len(seq))
        start = len(seq)
        seq += ids
        if cfg.sink == "drop" and len(ids) > 1:
            start += 1                      # first real token acts as the sink, skip it
        return seq, (start, len(seq))

    def collate(
        self,
        ids_list: Sequence[list[int]],
        spans: Sequence[tuple[int, int]],
        idf: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = len(ids_list)
        max_len = max(len(x) for x in ids_list)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.long)
        pool_w = torch.zeros((bsz, max_len), dtype=torch.float32)
        span_mask = torch.zeros((bsz, max_len), dtype=torch.bool)

        for i, (ids, (a, b)) in enumerate(zip(ids_list, spans)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attn[i, :n] = 1
            span_mask[i, a:b] = True
            pool_w[i, a:b] = 1.0

        if idf is not None:
            pool_w = pool_w * idf[input_ids.clamp(max=idf.numel() - 1)]

        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "position_ids": pos.unsqueeze(0).expand(3, -1, -1).contiguous(),  # M-RoPE (3,B,T)
            "pool_w": pool_w,
            "span_mask": span_mask,
        }


# --------------------------------------------------------------------------- #
# Pooling: mean + signed abs-max, each L2-normalised, concatenated  (2*D dims)
# h: [B,T,D] float, w: [B,T] weights (already zero outside mask), mask: [B,T] bool
# --------------------------------------------------------------------------- #
def _signed_abs_max(h_eff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per dimension, pick the token with the largest |value| and keep its sign."""
    mag = h_eff.abs().masked_fill(~mask.unsqueeze(-1), -1.0)
    idx = mag.argmax(dim=1, keepdim=True)  # [B,1,D]
    return h_eff.gather(dim=1, index=idx).squeeze(1)


def pool_mean_max(h: torch.Tensor, w: torch.Tensor, mask: torch.Tensor, use_idf: bool) -> torch.Tensor:
    ww = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    mean_vec = F.normalize((h * ww.unsqueeze(-1)).sum(dim=1), p=2, dim=-1)
    h_eff = h * w.clamp(min=0.0).sqrt().unsqueeze(-1) if use_idf else h
    amax_vec = F.normalize(_signed_abs_max(h_eff, mask), p=2, dim=-1)
    return torch.cat([mean_vec, amax_vec], dim=-1)


# --------------------------------------------------------------------------- #
# Post-processing: centre -> standardise -> remove top PCs -> soft whitening -> re-weight halves -> L2
# --------------------------------------------------------------------------- #
class PostProcessor:
    def __init__(
        self,
        standardize: bool = True,
        remove_pcs: int = 4,
        zca_alpha: float = 0.5,
        zca_rank: int = 128,
        eps: float = 1e-5,
        separate_query_std: bool = False,
        half_weight: float = 1.0,
        mean_weight: float = 1.0,  # Added mean_weight parameter
    ):
        self.standardize = standardize
        self.remove_pcs = remove_pcs
        self.zca_alpha = zca_alpha
        self.zca_rank = zca_rank
        self.eps = eps
        self.separate_query_std = separate_query_std
        self.half_weight = half_weight
        self.mean_weight = mean_weight
        self.mean_doc: torch.Tensor | None = None
        self.mean_query: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.std_query: torch.Tensor | None = None
        self.v_pcs: torch.Tensor | None = None  # [L, D, q]
        self.svals: torch.Tensor | None = None  # [L, q]

    def fit(self, docs_raw: torch.Tensor, queries_raw: torch.Tensor | None = None) -> "PostProcessor":
        x = docs_raw.float()  # [N, L, D]
        n, n_layers, d = x.shape
        self.mean_doc = x.mean(0)
        self.mean_query = queries_raw.float().mean(0) if queries_raw is not None else self.mean_doc
        if self.standardize and n > 1:
            self.std = x.std(0).clamp(min=self.eps)
        else:
            self.std = torch.ones_like(self.mean_doc)

        # Queries are short, docs are long: the max-pool half has very different per-dim variance.
        self.std_query = None
        if (
            self.separate_query_std and self.standardize
            and queries_raw is not None and queries_raw.shape[0] > 1
        ):
            self.std_query = queries_raw.float().std(0).clamp(min=self.eps)

        self.v_pcs = None
        self.svals = None
        # q <= N/2 keeps the reference singular value s[-1] well above the noise floor of a small corpus.
        q = min(max(self.zca_rank, self.remove_pcs), max(n // 2, 2), n - 1, d)
        if (self.remove_pcs > 0 or self.zca_alpha > 0) and q >= 2:
            vs, ss = [], []
            for l in range(n_layers):
                z = (x[:, l] - self.mean_doc[l]) / self.std[l]
                if n <= 8192:
                    _, s, vh = torch.linalg.svd(z, full_matrices=False)
                    v, s = vh[:q].T.contiguous(), s[:q]
                else:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(0)
                        _, s, v = torch.svd_lowrank(z, q=q, niter=4)
                vs.append(v)
                ss.append(s)
            self.v_pcs = torch.stack(vs, dim=0)
            self.svals = torch.stack(ss, dim=0)
        return self

    def transform(self, raw: torch.Tensor, role: str = "doc") -> torch.Tensor:
        z = raw.float()
        if self.mean_doc is not None:
            mean = self.mean_query if role == "query" else self.mean_doc
            std = self.std_query if (role == "query" and self.std_query is not None) else self.std
            z = (z - mean) / std
            if self.v_pcs is not None and self.svals is not None:
                v = self.v_pcs
                if self.remove_pcs > 0:
                    vt = v[:, :, : min(self.remove_pcs, v.size(2))]
                    z = z - torch.einsum("nlk,ldk->nld", torch.einsum("nld,ldk->nlk", z, vt), vt)
                if self.zca_alpha > 0:
                    # Rescale ONLY the top-q subspace; the residual is left untouched.
                    # scale == 1 at the edge of the subspace and < 1 for dominant directions.
                    scale = (self.svals / self.svals[:, -1:].clamp(min=self.eps)).pow(-self.zca_alpha)
                    coef = torch.einsum("nld,ldq->nlq", z, v)
                    z = z + torch.einsum("nlq,ldq->nld", coef * (scale.unsqueeze(0) - 1.0), v)
        
        # Updated to apply mean_weight to the first half (mean-pooled representation)
        if self.mean_weight != 1.0 or self.half_weight != 1.0:
            half = z.size(-1) // 2
            z = torch.cat([z[..., :half] * self.mean_weight, z[..., half:] * self.half_weight], dim=-1)
            
        z = F.normalize(z, p=2, dim=-1)
        return F.normalize(z.mean(dim=1), p=2, dim=-1)

    def state_dict(self) -> dict:
        return dict(
            standardize=self.standardize, remove_pcs=self.remove_pcs, zca_alpha=self.zca_alpha,
            zca_rank=self.zca_rank, mean_doc=self.mean_doc, mean_query=self.mean_query,
            std=self.std, std_query=self.std_query, v_pcs=self.v_pcs, svals=self.svals,
            separate_query_std=self.separate_query_std, half_weight=self.half_weight,
            mean_weight=self.mean_weight,
        )

    @classmethod
    def from_state(cls, s: dict) -> "PostProcessor":
        p = cls(s["standardize"], s["remove_pcs"], s.get("zca_alpha", 0.5), s.get("zca_rank", 128),
                separate_query_std=s.get("separate_query_std", False), half_weight=s.get("half_weight", 1.0),
                mean_weight=s.get("mean_weight", 1.0))
        p.mean_doc, p.mean_query, p.std = s["mean_doc"], s["mean_query"], s["std"]
        p.std_query = s.get("std_query")
        p.v_pcs, p.svals = s.get("v_pcs"), s.get("svals")
        return p


# --------------------------------------------------------------------------- #
# Metrics / data helpers
# --------------------------------------------------------------------------- #
def metrics_from_scores(
    scores: torch.Tensor,
    relevant: Sequence[Collection[int]],
    ks: Sequence[int] = (1, 5, 10),
    per_query: bool = False,
) -> dict:
    """acc@k, mrr, mrr@5, ndcg@10 (binary relevance). Values in [0, 1]."""
    scores = scores.float().cpu()
    n_q, n_d = scores.shape
    if len(relevant) != n_q:
        raise ValueError("`relevant` must have one entry per query.")
    order = scores.argsort(dim=1, descending=True)
    rel = torch.zeros(n_q, n_d, dtype=torch.bool)
    for i, r in enumerate(relevant):
        rel[i, torch.as_tensor(sorted(r), dtype=torch.long)] = True
    ranked = rel.gather(1, order)
    has = ranked.any(dim=1)
    first = ranked.float().argmax(dim=1) + 1  # 1-based rank of the best relevant doc
    zeros = torch.zeros(n_q)
    rr = torch.where(has, 1.0 / first.float(), zeros)
    mrr5 = torch.where(has & (first <= 5), 1.0 / first.float(), zeros)

    kk = min(10, n_d)
    disc = 1.0 / torch.log2(torch.arange(2, kk + 2, dtype=torch.float32))
    dcg = (ranked[:, :kk].float() * disc).sum(dim=1)
    n_rel = rel.sum(dim=1).clamp(min=1, max=kk)
    ndcg = dcg / disc.cumsum(dim=0)[n_rel - 1]

    out: dict = {f"acc@{k}": ((first <= min(k, n_d)) & has).float().mean().item() for k in ks}
    out["mrr"] = rr.mean().item()
    out["mrr@5"] = mrr5.mean().item()
    out["ndcg@10"] = ndcg.mean().item()
    if per_query:
        out["_ndcg_pq"] = ndcg
        out["_mrr5_pq"] = mrr5
    return out


def format_metrics(m: dict) -> str:
    return " | ".join(f"{k} = {100 * v:.2f}" for k, v in m.items() if not k.startswith("_"))


def bootstrap_ci(x: torch.Tensor, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean and (1-alpha) bootstrap CI over queries."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, x.numel(), (n_boot, x.numel()), generator=g)
    means = x[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return x.mean().item(), lo, hi


def split_queries(n: int, holdout: float = 0.5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic dev / held-out split shared by tune_post and tune_fusion."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_dev = n if holdout <= 0 else max(1, int(round(n * (1.0 - holdout))))
    return perm[:n_dev], perm[n_dev:]


def build_relevance(corpus_ids: Sequence[str], queries, qrels) -> tuple[list[str], list[str], list[set[int]]]:
    """(corpus ids, {qid: text} or [(qid, text)], {qid: {cid: grade}}) -> (qids, texts, relevant index sets).
    Keeps only queries that have at least one judged-relevant document."""
    cid2idx = {c: i for i, c in enumerate(corpus_ids)}
    items = list(queries.items()) if isinstance(queries, dict) else list(queries)
    qids, texts, relevant = [], [], []
    for qid, text in items:
        rel = {cid2idx[c] for c, g in qrels.get(qid, {}).items() if g > 0 and c in cid2idx}
        if rel:
            qids.append(qid)
            texts.append(text)
            relevant.append(rel)
    return qids, texts, relevant


# --------------------------------------------------------------------------- #
# Lexical channel: word-level BM25
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    "a an the of in on at to for and or but with without by from as is are was were be been being am "
    "do does did has have had having it its this that these those what which who whom whose when where "
    "why how can could should would will may might must shall i you he she we they me my your his her "
    "our their there here than then so such also about into over under between during after before "
    "above below up down out off again further once more most other some any each few both very just "
    "only own same too".split()
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _s_stem(w: str) -> str:
    """Harman's S-stemmer: plural stripping only (safe for medical vocabulary)."""
    n = len(w)
    if n > 4 and w.endswith("ies") and w[-4] not in "ae":
        return w[:-3] + "y"
    if n > 3 and w.endswith("es"):
        return w[:-1] if w[-3] not in "aeo" else w
    if n > 3 and w.endswith("s") and w[-2] not in "us":
        return w[:-1]
    return w


class LexicalIndex:
    """BM25 over lower-cased, stopword-filtered, S-stemmed words (optionally + adjacent bigrams).
    The doc-side BM25 weight does not depend on the query, so it is precomputed per term."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75, bigrams: bool = False):
        self.k1, self.b, self.bigrams = k1, b, bigrams
        docs = [self.terms(t) for t in texts]
        self.n_docs = len(docs)
        dl = torch.tensor([max(len(d), 1) for d in docs], dtype=torch.float32)
        avg = float(dl.mean()) if self.n_docs else 1.0
        by_term: dict[str, list[tuple[int, int]]] = {}
        for j, d in enumerate(docs):
            for t, f in Counter(d).items():
                by_term.setdefault(t, []).append((j, f))
        self.post: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for t, lst in by_term.items():
            idx = torch.tensor([j for j, _ in lst], dtype=torch.long)
            tf = torch.tensor([f for _, f in lst], dtype=torch.float32)
            n = len(lst)
            idf = math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            norm = k1 * (1.0 - b + b * dl[idx] / avg)
            self.post[t] = (idx, idf * tf * (k1 + 1.0) / (tf + norm))

    def terms(self, text: str) -> list[str]:
        toks = _TOKEN_RE.findall(unicodedata.normalize("NFKC", str(text)).lower())
        toks = [_s_stem(t) for t in toks if t not in _STOPWORDS]
        if self.bigrams:
            toks = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        return toks

    def score(self, queries: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(queries), self.n_docs)
        for i, q in enumerate(queries):
            for t in set(self.terms(q)):
                p = self.post.get(t)
                if p is not None:
                    out[i].index_add_(0, p[0], p[1])
        return out


# --------------------------------------------------------------------------- #
# Fusion, pseudo-relevance feedback, Sinkhorn
# --------------------------------------------------------------------------- #
def _zs(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp(min=1e-6)


def fuse_scores(dense: torch.Tensor, sparse: torch.Tensor, w: float) -> torch.Tensor:
    """(1-w) * z(dense) + w * z(sparse), z-scored per query. w<=0 returns `dense` untouched."""
    if w <= 0:
        return dense
    return (1.0 - w) * _zs(dense) + w * _zs(sparse)


def rocchio(q: torch.Tensor, d: torch.Tensor, scores: torch.Tensor, k: int, beta: float, chunk: int = 256):
    """q' = normalise(q + beta * normalise(mean of the top-k docs under `scores`))."""
    k = max(1, min(k, d.size(0)))
    top = scores.topk(k, dim=1).indices
    out = []
    for s in range(0, q.size(0), chunk):
        cent = F.normalize(d[top[s:s + chunk]].mean(dim=1), p=2, dim=-1)
        out.append(F.normalize(q[s:s + chunk] + beta * cent, p=2, dim=-1))
    return torch.cat(out)


_MIN_Q_FOR_NORM = 16   # query-set normalisation collapses to ties for a handful of queries


def sinkhorn_scores(s: torch.Tensor, tau: float, iters: int = 1) -> torch.Tensor:
    """Query-set-aware normalisation of a [Q, N] score matrix (inverted softmax / Sinkhorn).

    Each document can only be the best answer to a few queries. Normalising every column over all queries
    demotes documents that many other queries already claim. TRANSDUCTIVE: needs the whole query batch."""
    z = (s - s.mean()) / s.std().clamp(min=1e-6)
    lp = z / tau
    for i in range(max(iters, 1)):
        if i > 0:
            lp = lp - torch.logsumexp(lp, dim=1, keepdim=True)    # each query spreads unit mass over docs
        lp = lp - torch.logsumexp(lp, dim=0, keepdim=True)        # each doc spreads unit mass over queries
    return lp


# --------------------------------------------------------------------------- #
# Embedder / retriever
# --------------------------------------------------------------------------- #
class MedicalTextEmbedder:
    # config fields restored by load_index so a saved index reproduces the tuned pipeline
    _TUNED = (
        "hybrid_weight", "prf_k", "prf_beta", "hub_gamma", "norm_tau", "norm_iters",
        "standardize", "separate_query_std", "remove_pcs", "zca_alpha", "zca_rank", "half_weight", "mean_weight",
        "bm25_k1", "bm25_b", "lex_bigrams",
    )
    _CFG_KEYS = ("layers", "echo", "sink", "token_zscore", "mask_outlier_tokens", "token_norm", "use_idf")

    def __init__(self, config: EmbedderConfig | None = None, model=None, tokenizer=None):
        self.config = config or EmbedderConfig()
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, trust_remote_code=True)
        if model is None:
            model = Qwen3VLTextModel.from_pretrained(
                self.config.model_dir,
                device_map="auto",
                dtype=self.config.dtype,
                trust_remote_code=True,
            )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.builder = PromptBuilder(self.tokenizer, self.config)
        self.idf: torch.Tensor | None = None
        self.tok_stats: dict[int, dict] = {}
        self.post: PostProcessor | None = None
        self.corpus_embeddings: torch.Tensor | None = None
        self.corpus_texts: list[str] | None = None
        self.corpus_ids: list[list[int]] | None = None
        self.lex: LexicalIndex | None = None
        self.hub_r: torch.Tensor | None = None
        self.last_tune: dict = {}
        self._raw_docs: torch.Tensor | None = None
        self._raw_fit_q: torch.Tensor | None = None
        self._raw_q: dict[str, torch.Tensor] = {}
        self._warned = False

    # ---- small helpers ------------------------------------------------------
    @staticmethod
    def _validate(texts: Sequence[str]) -> None:
        if isinstance(texts, str):
            raise TypeError("Expected a sequence of strings, not a single string.")
        if len(texts) == 0:
            raise ValueError("Input texts cannot be empty.")
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Every input must be a non-empty string.")

    def _dim(self) -> int:
        return self.model.config.hidden_size * 2          # mean half + max half

    @property
    def embedding_dimension(self) -> int:
        return self._dim()

    def _norm_layers(self, layers: Sequence[int]) -> tuple[int, ...]:
        total = len(self.model.layers) + 1
        out = []
        for l in layers:
            if not -total <= l < total:
                raise ValueError(f"layer {l} invalid: model has hidden states 0..{total - 1}.")
            out.append(l + total if l < 0 else l)
        return tuple(out)

    @contextmanager
    def _truncated(self, max_layer: int) -> Iterator[None]:
        """Run only the first max_layer+1 decoder layers. Entries 0..max_layer of hidden_states are
        identical to the full model (only the final entry is norm-ed, and we never read it)."""
        full = self.model.layers
        keep = min(len(full), max_layer + 1)
        if not self.config.truncate_layers or keep >= len(full):
            yield
            return
        self.model.layers = full[:keep]
        try:
            yield
        finally:
            self.model.layers = full

    @staticmethod
    def _batches(order: Sequence[int], lengths: Sequence[int], bs: int, cap: int) -> Iterator[list[int]]:
        cur: list[int] = []
        mx = 0
        for i in order:  # ascending length -> minimal padding
            new_mx = max(mx, lengths[i])
            if cur and (len(cur) >= bs or new_mx * (len(cur) + 1) > cap):
                yield cur
                cur, new_mx = [], lengths[i]
            cur.append(i)
            mx = new_mx
        if cur:
            yield cur

    def _forward(self, batch: dict[str, torch.Tensor]):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states

    def _prepare(self, texts: Sequence[str]):
        prepared = [self.builder.build(t) for t in texts]
        lengths = [len(p[0]) for p in prepared]
        order = sorted(range(len(texts)), key=lengths.__getitem__)
        return prepared, lengths, order

    def _make_batch(self, prepared, idx, idf) -> dict[str, torch.Tensor]:
        batch = self.builder.collate([prepared[i][0] for i in idx], [prepared[i][1] for i in idx], idf)
        device = self.model.device
        return {k: v.to(device) for k, v in batch.items()}

    # ---- IDF + token-level calibration (label-free) -------------------------------
    def fit_idf(self, corpus_texts: Sequence[str]) -> None:
        self._validate(corpus_texts)
        vocab = self.model.get_input_embeddings().weight.size(0)
        df = torch.zeros(vocab)
        for t in corpus_texts:
            df[torch.tensor(sorted(set(self.builder.text_ids(t))))] += 1
        n = len(corpus_texts)
        self.idf = torch.log((n + 1.0) / (df + 1.0)) + 1.0

    @torch.inference_mode()
    def calibrate(self, texts: Sequence[str], batch_size: int | None = None) -> None:
        """Estimate per-layer token statistics (mean/std per dim, median norm) from unlabeled text."""
        self._validate(texts)
        cfg = self.config
        layers = self._norm_layers(cfg.layers)
        bs = batch_size or cfg.batch_size
        prepared, lengths, order = self._prepare(texts)
        d = self.model.config.hidden_size
        s1 = {l: torch.zeros(d, dtype=torch.float64) for l in layers}
        s2 = {l: torch.zeros(d, dtype=torch.float64) for l in layers}
        cnt = {l: 0 for l in layers}
        nrm: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

        with self._truncated(max(layers)):
            for idx in self._batches(order, lengths, bs, cfg.max_batch_tokens):
                batch = self._make_batch(prepared, idx, None)
                hs = self._forward(batch)
                m = batch["span_mask"]
                for l in layers:
                    h = hs[l].float()
                    norms = h.norm(dim=-1)
                    med = float(norms[m].cpu().median())
                    keep = m & (norms <= cfg.outlier_ratio * med)
                    n_keep = int(keep.sum())
                    if n_keep == 0:
                        continue
                    hk = h[keep]
                    s1[l] += hk.sum(0).cpu().double()
                    s2[l] += (hk * hk).sum(0).cpu().double()
                    cnt[l] += n_keep
                    nrm[l].append(norms[keep].cpu())

        device = self.model.device
        for l in layers:
            if cnt[l] == 0:
                continue
            mean = s1[l] / cnt[l]
            std = (s2[l] / cnt[l] - mean ** 2).clamp(min=0).sqrt()
            std = std.clamp(min=float((0.1 * std.median()).clamp(min=1e-6)))
            self.tok_stats[l] = dict(
                mean=mean.float().to(device),
                std=std.float().to(device),
                norm_med=float(torch.cat(nrm[l]).median()),
            )

    def _prep_layer(self, h: torch.Tensor, batch: dict, layer: int):
        cfg = self.config
        h = h.float()
        mask = batch["span_mask"]
        st = self.tok_stats.get(layer)
        if st is not None:
            if cfg.mask_outlier_tokens:
                keep = h.norm(dim=-1) <= cfg.outlier_ratio * st["norm_med"]
                m2 = mask & keep
                m2 = torch.where(m2.any(dim=1, keepdim=True), m2, mask)  # never leave a row empty
            else:
                m2 = mask
            if cfg.token_zscore:
                h = (h - st["mean"]) / st["std"]
        else:
            m2 = mask
            if (cfg.token_zscore or cfg.mask_outlier_tokens) and not self._warned:
                warnings.warn(
                    f"No token statistics for layer {layer}: token_zscore / mask_outlier_tokens are skipped. "
                    "Call calibrate(texts) (index_corpus does this for you)."
                )
                self._warned = True
        if cfg.token_norm:
            h = F.normalize(h, p=2, dim=-1)
        return h, m2, batch["pool_w"] * m2.float()

    # ---- encoding -----------------------------------------------------------------
    @torch.inference_mode()
    def encode_raw(self, texts: Sequence[str], role: str = "doc", batch_size: int | None = None) -> torch.Tensor:
        """Raw pooled states, [N, n_layers, 2*D]. `role` is accepted for API symmetry."""
        self._validate(texts)
        cfg = self.config
        layers = self._norm_layers(cfg.layers)
        bs = batch_size or cfg.batch_size
        idf = self.idf if (cfg.use_idf and self.idf is not None) else None
        prepared, lengths, order = self._prepare(texts)
        out = torch.empty(len(texts), len(layers), self._dim())
        with self._truncated(max(layers)):
            for idx in self._batches(order, lengths, bs, cfg.max_batch_tokens):
                batch = self._make_batch(prepared, idx, idf)
                hs = self._forward(batch)
                sel = torch.tensor(idx)
                for li, l in enumerate(layers):
                    h, mk, w = self._prep_layer(hs[l], batch, l)
                    out[sel, li] = pool_mean_max(h, w, mk, cfg.use_idf).cpu()
        return out

    def _finish(self, raw: torch.Tensor, role: str) -> torch.Tensor:
        post = self.post if self.post is not None else PostProcessor(zca_alpha=0.0, remove_pcs=0)
        return post.transform(raw, role)

    def _raw_queries(self, queries: Sequence[str], batch_size: int | None = None) -> torch.Tensor:
        """Raw query states with a text-keyed cache, so tuning / evaluation / search never re-run the LLM
        on a query it has already seen."""
        self._validate(queries)
        miss = [q for q in dict.fromkeys(queries) if q not in self._raw_q]
        if miss:
            raw = self.encode_raw(miss, "query", batch_size)
            for q, r in zip(miss, raw):
                self._raw_q[q] = r
        return torch.stack([self._raw_q[q] for q in queries])

    def encode_documents(self, texts: Sequence[str], batch_size: int | None = None) -> torch.Tensor:
        return self._finish(self.encode_raw(texts, "doc", batch_size), "doc")

    def encode_queries(self, queries: Sequence[str], batch_size: int | None = None) -> torch.Tensor:
        return self._finish(self._raw_queries(list(queries), batch_size), "query")

    # ---- indexing ------------------------------------------------------------------------
    def index_corpus(
        self,
        corpus_texts: Sequence[str],
        fit_queries: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Encode the corpus, fit the post-processor, build the BM25 index and the hubness penalty.
        `fit_queries` are used label-free (calibration, query centering / std, hub reference set)."""
        self._validate(corpus_texts)
        cfg = self.config
        self._raw_q = {}                    # raw states depend on idf / calibration -> invalidate
        if cfg.use_idf:
            self.fit_idf(corpus_texts)
        if cfg.token_zscore or cfg.mask_outlier_tokens:
            self.calibrate(list(corpus_texts) + list(fit_queries or []), batch_size)
        raw_docs = self.encode_raw(corpus_texts, "doc", batch_size)
        raw_q = self._raw_queries(list(fit_queries), batch_size) if fit_queries else None
        self._raw_docs, self._raw_fit_q = raw_docs, raw_q
        self.corpus_texts = list(corpus_texts)
        self.corpus_ids = [self.builder.text_ids(t) for t in corpus_texts]
        self.lex = LexicalIndex(self.corpus_texts, cfg.bm25_k1, cfg.bm25_b, cfg.lex_bigrams)
        return self.refit_post()

    def refit_post(self, **overrides) -> torch.Tensor:
        """Re-fit the PostProcessor from cached raw states (no LLM calls). Accepts any of
        standardize / remove_pcs / zca_alpha / zca_rank / separate_query_std / half_weight / mean_weight and stores them
        in the config."""
        if self._raw_docs is None:
            raise RuntimeError("No cached raw embeddings: call index_corpus() first (load_index() does not keep them).")
        cfg = self.config
        allowed = ("standardize", "remove_pcs", "zca_alpha", "zca_rank", "separate_query_std", "half_weight", "mean_weight")
        for k, v in overrides.items():
            if k not in allowed:
                raise ValueError(f"Unknown post-processing option: {k}")
            setattr(cfg, k, v)
        self.post = PostProcessor(
            cfg.standardize, cfg.remove_pcs, cfg.zca_alpha, cfg.zca_rank,
            separate_query_std=cfg.separate_query_std, half_weight=cfg.half_weight,
            mean_weight=cfg.mean_weight,
        ).fit(self._raw_docs, self._raw_fit_q)
        self.corpus_embeddings = self.post.transform(self._raw_docs, "doc")
        self._build_hub()
        return self.corpus_embeddings

    def _build_hub(self) -> None:
        """r_d = mean cosine of doc d to its k nearest queries (or to its k nearest other docs when no
        queries were given). Subtracting gamma * r_d demotes hubs (CSLS)."""
        cfg = self.config
        self.hub_r = None
        if cfg.hub_k <= 0 or self.corpus_embeddings is None or self.post is None:
            return
        d = self.corpus_embeddings
        self_ref = self._raw_fit_q is None
        ref = d if self_ref else self.post.transform(self._raw_fit_q, "query")
        sims = d @ ref.T
        if self_ref:
            sims.fill_diagonal_(float("-inf"))
        k = min(cfg.hub_k, sims.size(1) - (1 if self_ref else 0))
        if k >= 1:
            self.hub_r = sims.topk(k, dim=1).values.mean(dim=1)

    # ---- scoring ---------------------------------------------------------------------------
    def _need_index(self) -> None:
        if self.corpus_embeddings is None or self.corpus_texts is None:
            raise RuntimeError("Call index_corpus() or load_index() first.")

    def _combine(
        self,
        qe: torch.Tensor,
        sp: torch.Tensor | None,
        w: float,
        prf_k: int,
        prf_beta: float,
        gamma: float,
        tau: float = 0.0,
        iters: int = 1,
    ) -> torch.Tensor:
        """dense (+hub penalty) -> fuse BM25 -> [Sinkhorn] -> optional Rocchio re-query -> fuse -> [Sinkhorn]."""
        d = self.corpus_embeddings
        pen = self.hub_r if (gamma > 0 and self.hub_r is not None) else None

        def dense(q: torch.Tensor) -> torch.Tensor:
            s = q @ d.T
            return s - gamma * pen if pen is not None else s

        use_sp = sp is not None and w > 0

        def finish(x: torch.Tensor) -> torch.Tensor:
            fused = fuse_scores(x, sp, w) if use_sp else x
            if tau > 0 and fused.size(0) >= _MIN_Q_FOR_NORM:
                fused = sinkhorn_scores(fused, tau, iters)
            return fused

        fused = finish(dense(qe))
        if prf_k > 0 and prf_beta > 0:
            fused = finish(dense(rocchio(qe, d, fused, prf_k, prf_beta)))
        return fused

    def _score_matrix(
        self,
        queries: Sequence[str],
        hybrid_weight: float | None = None,
        prf_k: int | None = None,
        prf_beta: float | None = None,
        hub_gamma: float | None = None,
        norm_tau: float | None = None,
        norm_iters: int | None = None,
    ) -> torch.Tensor:
        self._need_index()
        cfg = self.config
        queries = list(queries)
        w = cfg.hybrid_weight if hybrid_weight is None else hybrid_weight
        sp = self.lex.score(queries) if w > 0 else None
        return self._combine(
            self.encode_queries(queries), sp, w,
            cfg.prf_k if prf_k is None else prf_k,
            cfg.prf_beta if prf_beta is None else prf_beta,
            cfg.hub_gamma if hub_gamma is None else hub_gamma,
            cfg.norm_tau if norm_tau is None else norm_tau,
            cfg.norm_iters if norm_iters is None else norm_iters,
        )

    def search(self, query: str, top_k: int = 4) -> list[tuple[float, str]]:
        """Single query. Query-set normalisation is skipped automatically for fewer than 16 queries."""
        return self.search_batch([query], top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int = 4) -> list[list[tuple[float, str]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        scores = self._score_matrix(queries)
        top = torch.topk(scores, k=min(top_k, len(self.corpus_texts)), dim=1)
        return [
            [(float(s), self.corpus_texts[int(i)]) for s, i in zip(srow, irow)]
            for srow, irow in zip(top.values, top.indices)
        ]

    def evaluate(
        self,
        queries: Sequence[str],
        relevant: Sequence[Collection[int]],
        ks: Sequence[int] = (1, 5, 10),
        **overrides,
    ) -> dict:
        """Metrics of the full current pipeline (use overrides such as hybrid_weight=0.0 for ablations)."""
        self._need_index()
        return metrics_from_scores(self._score_matrix(queries, **overrides), relevant, ks)

    # ---- tuning (labels are used only to pick a few scalars, on a dev split) -------------------------
    def tune_post(
        self,
        queries: Sequence[str],
        relevant: Sequence[Collection[int]],
        holdout: float = 0.5,
        seed: int = 0,
        grid: dict | None = None,
        apply: bool = True,
    ) -> dict:
        """One coordinate-descent pass over the post-processing settings, scored by DENSE-only nDCG@10 on the
        dev split. Re-uses cached raw states, so it makes no LLM calls (each step is one SVD per layer)."""
        self._need_index()
        cfg = self.config
        queries = list(queries)
        dev = split_queries(len(queries), holdout, seed)[0].tolist()
        dev_q, dev_rel = [queries[i] for i in dev], [relevant[i] for i in dev]
        grid = grid or dict(
            remove_pcs=(2, 4, 8),
            zca_alpha=(0.3, 0.5, 0.7),
            half_weight=(0.7, 1.0, 1.4),
            mean_weight=(0.0, 0.5, 1.0), # Added mean_weight to tuning grid
            separate_query_std=(False, True),
        )
        start = {k: getattr(cfg, k) for k in grid}

        def dev_ndcg() -> float:
            s = self._score_matrix(dev_q, hybrid_weight=0.0, prf_k=0, prf_beta=0.0, hub_gamma=0.0, norm_tau=0.0)
            return metrics_from_scores(s, dev_rel)["ndcg@10"]

        best_all = dev_ndcg()
        print(f"[tune_post] {len(dev)} dev queries | start dense nDCG@10 {100 * best_all:.2f} {start}")
        for name, values in grid.items():
            cur = getattr(cfg, name)
            best_v, best_s = cur, best_all
            for v in values:
                if v == cur:
                    continue
                self.refit_post(**{name: v})
                sc = dev_ndcg()
                print(f"  {name}={v!s:<6} dev dense nDCG@10 {100 * sc:5.2f}")
                if sc > best_s + 1e-4:
                    best_v, best_s = v, sc
            if getattr(cfg, name) != best_v:
                self.refit_post(**{name: best_v})
            best_all = best_s
        final = {k: getattr(cfg, k) for k in grid}
        if not apply:
            self.refit_post(**start)
            final = start
        print(f"[tune_post] {'applied' if apply else 'kept start'}: {final} | dev dense nDCG@10 {100 * best_all:.2f}")
        return final

    def tune_fusion(
        self,
        queries: Sequence[str],
        relevant: Sequence[Collection[int]],
        weights: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.5),
        prf: Sequence[tuple[int, float]] = (
            (0, 0.0), (3, 0.3), (3, 0.6), (5, 0.5), (5, 1.0), (10, 0.5), (10, 1.0)),
        gammas: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
        norms: Sequence[tuple[float, int]] = (
            (0.3, 1), (0.5, 1), (1.0, 1), (0.5, 3), (1.0, 3), (2.0, 3)),
        holdout: float = 0.5,
        seed: int = 0,
        top: int = 8,
        apply: bool = True,
    ) -> list[dict]:
        """Grid over (BM25 weight, PRF, hub gamma), then a second stage over the Sinkhorn temperature, on the
        dev split. Prints held-out nDCG@10 / MRR@5 with bootstrap CIs (the numbers to trust) and, if `apply`,
        writes the winner to the config. Pass norms=() to switch the transductive Sinkhorn stage off."""
        self._need_index()
        queries = list(queries)
        qe = self.encode_queries(queries)
        sp = self.lex.score(queries)
        n_q = len(queries)
        dev_idx, test_idx = split_queries(n_q, holdout, seed)
        n_dev = int(dev_idx.numel())

        def run(w, k, beta, g, tau=0.0, it=1):
            s = self._combine(qe, sp, w, k, beta, g, tau, it)
            return metrics_from_scores(s, relevant, per_query=True)["_ndcg_pq"]

        def held(nd) -> str:
            if not len(test_idx):
                return ""
            m, lo, hi = bootstrap_ci(nd[test_idx])
            return f" | held-out {100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:4.1f}]"

        base = run(0.0, 0, 0.0, 0.0)
        rows = []
        for w in weights:
            for k, beta in prf:
                for g in gammas:
                    nd = run(w, k, beta, g)
                    rows.append(dict(
                        hybrid_weight=w, prf_k=k, prf_beta=beta, hub_gamma=g,
                        dev=nd[dev_idx].mean().item(),
                        test=nd[test_idx].mean().item() if len(test_idx) else float("nan"), nd=nd,
                    ))
        rows.sort(key=lambda r: r["dev"], reverse=True)
        print(f"[tune_fusion] {n_dev} dev / {len(test_idx)} held-out queries")
        print(f"  dense only          dev {100 * base[dev_idx].mean().item():5.1f}{held(base)}")
        for r in rows[:top]:
            print(f"  w={r['hybrid_weight']:<4} prf=({r['prf_k']},{r['prf_beta']}) gamma={r['hub_gamma']:<4}"
                  f" dev {100 * r['dev']:5.1f}{held(r['nd'])}")

        norm_best = (0.0, 1)                      # stage 2: Sinkhorn on top of the best stage-1 setting
        if rows and norms and n_q >= _MIN_Q_FOR_NORM:
            b = rows[0]
            print("  -- Sinkhorn normalisation (transductive) on top of the best setting --")
            cand = []
            for tau, it in norms:
                nd = run(b["hybrid_weight"], b["prf_k"], b["prf_beta"], b["hub_gamma"], tau, it)
                cand.append((nd[dev_idx].mean().item(), tau, it, nd))
            cand.sort(key=lambda c: c[0], reverse=True)
            for dv, tau, it, nd in cand[:3]:
                print(f"  tau={tau:<4} iters={it} dev {100 * dv:5.1f}{held(nd)}")
            if cand[0][0] > b["dev"]:
                norm_best = (cand[0][1], cand[0][2])

        if apply and rows:
            b = rows[0]
            cfg = self.config
            cfg.hybrid_weight, cfg.prf_k, cfg.prf_beta, cfg.hub_gamma = (
                b["hybrid_weight"], b["prf_k"], b["prf_beta"], b["hub_gamma"])
            cfg.norm_tau, cfg.norm_iters = norm_best
            print(f"[tune_fusion] applied: hybrid_weight={cfg.hybrid_weight} prf_k={cfg.prf_k} "
                  f"prf_beta={cfg.prf_beta} hub_gamma={cfg.hub_gamma} "
                  f"norm_tau={cfg.norm_tau} norm_iters={cfg.norm_iters}")

        if rows:
            b = rows[0]
            fin = metrics_from_scores(
                self._combine(qe, sp, b["hybrid_weight"], b["prf_k"], b["prf_beta"], b["hub_gamma"], *norm_best),
                relevant, per_query=True)
            nd_f, mr_f = fin["_ndcg_pq"], fin["_mrr5_pq"]
            self.last_tune = dict(
                ndcg_dev=nd_f[dev_idx].mean().item(), mrr5_dev=mr_f[dev_idx].mean().item(),
                n_dev=n_dev, n_test=int(len(test_idx)),
            )
            msg = (f"[tune_fusion] FINAL dev nDCG@10 {100 * self.last_tune['ndcg_dev']:5.1f} "
                   f"MRR@5 {100 * self.last_tune['mrr5_dev']:5.1f}")
            if len(test_idx):
                m, lo, hi = bootstrap_ci(nd_f[test_idx])
                mm, mlo, mhi = bootstrap_ci(mr_f[test_idx])
                self.last_tune.update(ndcg_test=m, ndcg_lo=lo, ndcg_hi=hi, mrr5_test=mm, mrr5_lo=mlo, mrr5_hi=mhi)
                msg += (f" | HELD-OUT nDCG@10 {100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:4.1f}]"
                        f" MRR@5 {100 * mm:5.1f} [{100 * mlo:4.1f},{100 * mhi:4.1f}]")
            print(msg)
        return rows

    # ---- persistence ----------------------------------------------------------------
    def save_index(self, path: str) -> None:
        """Saves everything needed to serve queries (embeddings, post-processor, token stats, tuned scalars)."""
        if self.corpus_embeddings is None or self.post is None:
            raise RuntimeError("Nothing to save: call index_corpus() first.")
        torch.save(
            dict(
                cfg={k: getattr(self.config, k) for k in self._CFG_KEYS},
                tuned={k: getattr(self.config, k) for k in self._TUNED},
                corpus_texts=self.corpus_texts,
                corpus_embeddings=self.corpus_embeddings,
                idf=self.idf,
                hub_r=self.hub_r,
                post=self.post.state_dict(),
                tok_stats={l: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
                           for l, s in self.tok_stats.items()},
                # LongContextEmbedder fields
                original_texts=getattr(self, 'original_texts', None),
                doc_to_chunks=getattr(self, 'doc_to_chunks', None),
                chunk_size=getattr(self, 'chunk_size', None),
                chunk_overlap=getattr(self, 'chunk_overlap', None),
            ),
            path,
        )

    def load_index(self, path: str) -> None:
        s = torch.load(path, map_location="cpu", weights_only=True)
        for k, v in s["cfg"].items():
            cur = getattr(self.config, k)
            if (tuple(cur) if isinstance(cur, (list, tuple)) else cur) != (tuple(v) if isinstance(v, (list, tuple)) else v):
                raise ValueError(f"Index was built with {k}={v!r} but this embedder uses {k}={cur!r}.")
        for k, v in s.get("tuned", {}).items():
            setattr(self.config, k, v)
        cfg = self.config
        device = self.model.device
        self.corpus_texts = list(s["corpus_texts"])
        self.corpus_embeddings = s["corpus_embeddings"]
        self.idf = s["idf"]
        self.hub_r = s.get("hub_r")
        self.post = PostProcessor.from_state(s["post"])
        self.tok_stats = {
            int(l): {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
            for l, st in s["tok_stats"].items()
        }
        self.corpus_ids = [self.builder.text_ids(t) for t in self.corpus_texts]
        self.lex = LexicalIndex(self.corpus_texts, cfg.bm25_k1, cfg.bm25_b, cfg.lex_bigrams)
        self._raw_docs = self._raw_fit_q = None
        self._raw_q = {}
        
        # Load LongContextEmbedder fields if they exist
        self.original_texts = s.get("original_texts")
        self.doc_to_chunks = s.get("doc_to_chunks")
        if s.get("chunk_size") is not None:
            self.chunk_size = s.get("chunk_size")
        if s.get("chunk_overlap") is not None:
            self.chunk_overlap = s.get("chunk_overlap")


# --------------------------------------------------------------------------- #
# Long Context Extension
# --------------------------------------------------------------------------- #
class LongContextEmbedder(MedicalTextEmbedder):
    """
    Extension of MedicalTextEmbedder that handles long documents by chunking them 
    and using MaxSim aggregation during search and evaluation.
    """
    def __init__(self, *args, chunk_size: int = 512, chunk_overlap: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.original_texts: list[str] = []
        self.doc_to_chunks: dict[int, list[int]] = {}

    def index_corpus(self, corpus_texts: Sequence[str], fit_queries: Sequence[str] | None = None, batch_size: int | None = None) -> torch.Tensor:
        self._validate(corpus_texts)
        self.original_texts = list(corpus_texts)
        self.chunk_texts = []
        self.doc_to_chunks = {}
        
        # 1. Chunking logic
        for doc_idx, text in enumerate(corpus_texts):
            ids = self.builder.text_ids(text)
            start = 0
            doc_chunks = []
            while start < len(ids):
                end = min(start + self.chunk_size, len(ids))
                chunk_ids = ids[start:end]
                # Decode back to text to feed into the underlying embedder which expects strings
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                self.chunk_texts.append(chunk_text)
                doc_chunks.append(len(self.chunk_texts) - 1)
                if end == len(ids):
                    break
                start += self.chunk_size - self.chunk_overlap
            self.doc_to_chunks[doc_idx] = doc_chunks
            
        # 2. Index the chunks using the parent class
        return super().index_corpus(self.chunk_texts, fit_queries, batch_size)

    def search(self, query: str, top_k: int = 4) -> list[tuple[float, str]]:
        chunk_scores = self._score_matrix([query])[0]
        doc_scores = torch.full((len(self.original_texts),), float('-inf'))
        
        # MaxSim: Take the maximum score among all chunks of a document
        for doc_idx, chunk_indices in self.doc_to_chunks.items():
            if chunk_indices:
                doc_scores[doc_idx] = chunk_scores[chunk_indices].max()
                
        top_docs = torch.topk(doc_scores, k=min(top_k, len(doc_scores)))
        return [(float(s), self.original_texts[int(i)]) for s, i in zip(top_docs.values, top_docs.indices)]

    def search_batch(self, queries: Sequence[str], top_k: int = 4) -> list[list[tuple[float, str]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        chunk_scores = self._score_matrix(queries)
        all_results = []
        for q_idx in range(len(queries)):
            doc_scores = torch.full((len(self.original_texts),), float('-inf'))
            for doc_idx, chunk_indices in self.doc_to_chunks.items():
                if chunk_indices:
                    doc_scores[doc_idx] = chunk_scores[q_idx, chunk_indices].max()
            top_docs = torch.topk(doc_scores, k=min(top_k, len(doc_scores)))
            results = [(float(s), self.original_texts[int(i)]) for s, i in zip(top_docs.values, top_docs.indices)]
            all_results.append(results)
        return all_results

    def evaluate(
        self,
        queries: Sequence[str],
        relevant: Sequence[Collection[int]],
        ks: Sequence[int] = (1, 5, 10),
        **overrides,
    ) -> dict:
        """Metrics of the full current pipeline using MaxSim chunk aggregation."""
        self._need_index()
        # Compute the score matrix over chunks, then aggregate to documents
        chunk_scores = self._score_matrix(queries, **overrides)
        n_q = len(queries)
        n_d = len(self.original_texts)
        doc_scores = torch.full((n_q, n_d), float('-inf'))
        
        for doc_idx, chunk_indices in self.doc_to_chunks.items():
            if chunk_indices:
                doc_scores[:, doc_idx] = chunk_scores[:, chunk_indices].max(dim=1).values
                
        return metrics_from_scores(doc_scores, relevant, ks)

