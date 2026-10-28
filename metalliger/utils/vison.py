"""
Vision and Multimodal Graph RAG Helper Module.

This module provides helper utilities and retrieval-augmented generation (Graph RAG)
for vision-language tasks using Qwen3-VL models. It includes:
  - Batch image processing and ViT embedding extraction using Qwen3-VL's block-diagonal grid attention.
  - Anisotropy correction and CSLS (Cross-Domain Similarity Local Scaling) hubness suppression.
  - Mutual k-NN similarity graph construction.
  - Sparse graph operator precomputation (row-stochastic smoothing matrix and PageRank transition).
  - Vectorized Personalized PageRank (PPR) for clinical subgraph retrieval.

QUERY ENCODING is delegated entirely to a text.py `TextEmbedder` passed as `embedder=`:
  - the embedder supplies the model, tokenizer, token conditioning (z-score, outlier masking, token norm)
    and mean pooling; `layers=` picks which hidden states form the query (0 = token embeddings)
  - only the mean-pooled half of its output is used (image vectors are mean-pooled)
  - the query side is centred with its own per-layer mean (the image side with the image mean)
  - stopwords can be dropped from the query, and an optional BM25 channel over node text labels is fused
  - this module no longer loads or touches a Qwen3VLTextModel / tokenizer itself
"""

import hashlib
import logging
import re
import string
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLVisionModel

# Configure logger for Clinical Graph RAG operations
logger = logging.getLogger("UNClinicalGraphRAG")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Default number of neighboring nodes retrieved during Graph RAG traversal
K_NEIGHBORS = 10

# Global engine cache reference for lazy initialization
_CACHED_ENGINE: "UNClinicalGraphRAGEngine | None" = None


# --------------------------------------------------------------------------- #
# Lexical helpers (same normalisation as the text embedder)
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

# Generic, label-free imaging phrases used ONLY to estimate the mean token state of the query domain.
# Pass `fit_queries=` to the engine to use your own domain's phrasing instead / in addition.
_DEFAULT_CAL_QUERIES = (
    "Chest X-ray", "CT scan of the brain", "MRI of the knee", "Radiograph of the foot",
    "Ultrasound of the abdomen", "Mammography of the breast", "Panoramic dental radiograph",
    "Fundus photograph of the retina", "Dermoscopy of a skin lesion", "Histopathology slide of tissue",
    "Echocardiogram of the heart", "MRI of the lumbar spine", "Pelvic radiograph",
    "Abdominal CT with contrast", "Radiograph of the hand", "MRI of the shoulder",
    "Lung nodule on CT", "Fracture of the wrist", "Whole body bone scan", "PET scan of the body",
    "Coronary angiography", "Endoscopy image of the stomach", "Ultrasound of the liver",
    "CT of the kidney", "Thyroid ultrasound", "Brain MRI with tumor", "Pneumonia on chest radiograph",
    "Radiograph of the ankle", "Cervical spine X-ray", "Retinal OCT scan", "Colonoscopy image",
    "Prostate MRI",
)


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


class _NodeLexicalIndex:
    """Small BM25 index over node text labels (lower-cased, stopword-filtered, S-stemmed, + bigrams)."""

    def __init__(self, texts: Sequence[str | None], k1: float = 1.2, b: float = 0.75):
        docs = [self.terms(t) if t else [] for t in texts]
        self.n_docs = len(docs)
        dl = np.array([max(len(d), 1) for d in docs], dtype=np.float32)
        avg = float(dl.mean()) if self.n_docs else 1.0
        by_term: dict[str, list[tuple[int, int]]] = {}
        for j, d in enumerate(docs):
            for t, f in Counter(d).items():
                by_term.setdefault(t, []).append((j, f))
        self.post: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t, lst in by_term.items():
            idx = np.array([j for j, _ in lst], dtype=np.int64)
            tf = np.array([f for _, f in lst], dtype=np.float32)
            n = len(lst)
            idf = np.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            norm = k1 * (1.0 - b + b * dl[idx] / avg)
            self.post[t] = (idx, (idf * tf * (k1 + 1.0) / (tf + norm)).astype(np.float32))

    @staticmethod
    def terms(text: str) -> list[str]:
        toks = _TOKEN_RE.findall(unicodedata.normalize("NFKC", str(text)).lower())
        toks = [_s_stem(t) for t in toks if t not in _STOPWORDS]
        return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]

    def score(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for t in set(self.terms(query)):
            p = self.post.get(t)
            if p is not None:
                out[p[0]] += p[1]
        return out


def _zs(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-6)


class func:
    """
    Multimodal Vision Helper Utilities.

    Provides utility methods for loading, resizing, and hashing images, batch
    encoding with Qwen3-VL vision backbones, and building calibrated mutual
    k-NN similarity graphs for retrieval.
    """

    @staticmethod
    def _load_and_hash_image(
        img_input: str | Image.Image, max_edge: int = 512
    ) -> tuple[Image.Image, str, str | None]:
        """
        Loads, resizes, and generates an audit hash for an image input.

        Args:
            img_input (str | Image.Image): File path string or PIL Image object.
            max_edge (int, optional): Maximum width/height in pixels (LANCZOS downscale). Defaults to 512.

        Returns:
            tuple: (rgb_img, first 16 hex chars of the SHA-256 of the resized pixels, source path or None).

        Raises:
            TypeError: If img_input is neither a string nor a PIL.Image instance.
        """
        source_path = img_input if isinstance(img_input, str) else getattr(img_input, "filename", None)

        if isinstance(img_input, str):
            with Image.open(img_input) as raw:
                rgb_img = raw.convert("RGB")
        elif isinstance(img_input, Image.Image):
            rgb_img = img_input.convert("RGB")
        else:
            raise TypeError(f"Expected file path or PIL.Image, got {type(img_input)}")

        w, h = rgb_img.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            rgb_img = rgb_img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
            )

        img_hash = hashlib.sha256(rgb_img.tobytes()).hexdigest()[:16]
        return rgb_img, img_hash, source_path

    @torch.inference_mode()
    def encode_images_batch(
        self,
        image_processor: AutoProcessor,
        vision_model: Qwen3VLVisionModel,
        pil_images: list[Image.Image],
    ) -> np.ndarray:
        """
        Encodes a batch of PIL images in ONE forward pass (block-diagonal `image_grid_thw` attention),
        mean-pools the merged patch embeddings per image and L2-normalises them.

        Returns:
            np.ndarray: [batch_size, hidden_dim] float32 on CPU.
        """
        device = next(vision_model.parameters()).device
        proc = getattr(image_processor, "image_processor", image_processor)
        inputs = proc(images=pil_images, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(device, dtype=vision_model.dtype, non_blocking=True)
        grid_thw = inputs["image_grid_thw"].to(device, non_blocking=True)

        out = vision_model(pixel_values, grid_thw=grid_thw, return_dict=True)
        # Newer transformers: ModelOutput with `pooler_output` = merged patches.
        # Older transformers: tuple whose first element is the merged patches.
        merged_patches = getattr(out, "pooler_output", None)
        if merged_patches is None:
            merged_patches = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state

        merge_factor = getattr(vision_model, "spatial_merge_size", 2) ** 2
        patches_per_img = (grid_thw.prod(dim=-1) // merge_factor).tolist()
        if merged_patches.size(0) != sum(patches_per_img):
            raise RuntimeError(
                f"Merged patch count {merged_patches.size(0)} != expected {sum(patches_per_img)}; "
                "the vision model output layout differs from what this helper expects."
            )

        split_patches = torch.split(merged_patches, patches_per_img, dim=0)
        pooled = torch.stack([p.mean(dim=0, dtype=torch.float32) for p in split_patches], dim=0)
        return F.normalize(pooled, p=2, dim=-1).cpu().numpy()

    def similarity_graph(
        self,
        images: Sequence[str | Image.Image],
        image_processor: AutoProcessor,
        vision_model: Qwen3VLVisionModel,
        texts: Sequence[str | None] | None = None,
        text_embeddings: np.ndarray | None = None,
        similarity_threshold: float = 0.85,
        batch_size: int = 16,
        max_neighbors_k: int = 12,
        mutual_knn: bool = True,
        calibrate_hubness: bool = True,
        hybrid_text_alpha: float = 0.35,
        store_raw_pil_in_nodes: bool = False,
    ) -> tuple[nx.Graph, np.ndarray, list[float]]:
        """
        Builds a mutual-kNN similarity graph from an image collection.

        Pipeline: validation -> micro-batched ViT embedding -> mean-centring + CSLS hubness suppression
        -> optional fusion with text embeddings -> vectorised (mutual) kNN + cosine threshold -> NetworkX.

        Args:
            images: image paths or PIL images.
            image_processor / vision_model: Qwen3-VL processor and vision tower.
            texts: optional per-image text labels (stored on nodes, also used for the BM25 channel).
            text_embeddings: optional [N, D] text embeddings aligned with `images`.
            similarity_threshold: minimum RAW cosine for an edge (0..1).
            batch_size: ViT micro-batch size.
            max_neighbors_k: top-k neighbours considered per node.
            mutual_knn: require bidirectional top-k membership.
            calibrate_hubness: mean-centring + CSLS (needs > 10 nodes).
            hybrid_text_alpha: weight of text similarity when `text_embeddings` is given.
            store_raw_pil_in_nodes: keep PIL objects on nodes (RAM heavy) instead of path strings.

        Returns:
            (G, image_embeddings [N, D], per-image latency in ms)
        """
        # 1. INPUT VALIDATION
        if not isinstance(images, (list, tuple)):
            raise TypeError("`images` must be a list or tuple of PIL Images or file paths.")
        if texts is not None and len(texts) != len(images):
            raise ValueError("`texts` and `images` must have the exact same length.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("`similarity_threshold` must be between 0.0 and 1.0.")
        if not images:
            raise ValueError("The `images` list is empty.")

        valid_records = []
        for idx, item in enumerate(images):
            if item is None:
                logger.warning(f"Skipping index {idx}: image is None.")
                continue
            valid_records.append((idx, item, texts[idx] if texts is not None else None))
        if not valid_records:
            raise ValueError("No valid images were found to process.")

        # 2. MICRO-BATCHED VISION EMBEDDING EXTRACTION
        embeddings_batches = []
        time_to_process: list[float] = []
        node_metadata = []

        total_valid = len(valid_records)
        dev = next(vision_model.parameters()).device
        logger.info(f"Encoding {total_valid} medical scans (batch_size={batch_size}) on {dev}...")

        for b_start in range(0, total_valid, batch_size):
            batch_slice = valid_records[b_start: b_start + batch_size]
            batch_pils, batch_meta = [], []

            t0 = time.perf_counter()
            for orig_idx, img_item, txt in batch_slice:
                pil_img, sha_hash, src_path = self._load_and_hash_image(img_item)
                batch_pils.append(pil_img)
                batch_meta.append({
                    "orig_idx": orig_idx,
                    "image_ref": pil_img if store_raw_pil_in_nodes else (src_path or f"scan_{orig_idx}"),
                    "sha256": sha_hash,
                    "text": txt,
                })

            batch_embs = self.encode_images_batch(image_processor, vision_model, batch_pils)
            per_img_ms = round((time.perf_counter() - t0) * 1000.0 / len(batch_pils), 2)

            embeddings_batches.append(batch_embs)
            time_to_process.extend([per_img_ms] * len(batch_pils))
            node_metadata.extend(batch_meta)

            if not store_raw_pil_in_nodes:
                for p in batch_pils:
                    p.close()

        image_embeddings = np.vstack(embeddings_batches)
        num_nodes = image_embeddings.shape[0]

        # 3. ANISOTROPY CALIBRATION & CSLS HUBNESS SUPPRESSION
        raw_sim_matrix = image_embeddings @ image_embeddings.T

        if calibrate_hubness and num_nodes > 10:
            mean_vec = image_embeddings.mean(axis=0, keepdims=True)
            centered = image_embeddings - mean_vec
            cal_embs = centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(min=1e-9)
            cal_sim = cal_embs @ cal_embs.T
            # FIX: exclude the self-similarity BEFORE taking the k nearest. The old slice
            # [-(k+1):-1] of an argpartition is not "the k best without self" (only the pivot is ordered).
            np.fill_diagonal(cal_sim, -np.inf)
            k_csls = min(10, num_nodes - 1)
            knn_mean_sim = np.partition(cal_sim, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
            csls_matrix = cal_sim - 0.5 * (knn_mean_sim[:, None] + knn_mean_sim[None, :])
            topological_affinity = 0.5 * raw_sim_matrix + 0.5 * csls_matrix
        else:
            topological_affinity = raw_sim_matrix.copy()

        # Optional text fusion. FIX: align rows with the surviving nodes (None images were skipped)
        # and warn instead of silently ignoring a shape mismatch.
        if text_embeddings is not None:
            te = np.asarray(text_embeddings, dtype=np.float32)
            if te.shape[0] == len(images):
                te = te[[m["orig_idx"] for m in node_metadata]]
            if te.shape[0] == num_nodes:
                te_c = te - te.mean(axis=0, keepdims=True)
                t_cal = te_c / np.linalg.norm(te_c, axis=1, keepdims=True).clip(min=1e-9)
                text_sim = t_cal @ t_cal.T
                topological_affinity = (
                    (1.0 - hybrid_text_alpha) * topological_affinity + hybrid_text_alpha * text_sim
                )
            else:
                logger.warning(
                    f"text_embeddings has {te.shape[0]} rows for {num_nodes} nodes; text fusion skipped."
                )

        np.fill_diagonal(topological_affinity, -np.inf)
        np.fill_diagonal(raw_sim_matrix, 0.0)

        # 4. VECTORIZED MUTUAL k-NN & THRESHOLD GRAPH CONSTRUCTION
        G = nx.Graph()
        for idx, meta in enumerate(node_metadata):
            G.add_node(
                idx,
                image=meta["image_ref"],
                text=meta["text"],
                sha256=meta["sha256"],
                embedding=image_embeddings[idx],
            )

        if num_nodes > 1:
            k_eff = min(max_neighbors_k, num_nodes - 1)
            top_k_indices = np.argpartition(topological_affinity, -k_eff, axis=1)[:, -k_eff:]

            knn_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
            knn_mask[np.arange(num_nodes)[:, None], top_k_indices] = True

            adj_mask = (knn_mask & knn_mask.T) if mutual_knn else (knn_mask | knn_mask.T)
            threshold_mask = raw_sim_matrix >= similarity_threshold
            final_edge_mask = np.triu(adj_mask & threshold_mask, k=1)

            u_idx, v_idx = np.nonzero(final_edge_mask)
            G.add_edges_from(
                (int(u), int(v), {"weight": float(w), "calibrated_affinity": float(cw)})
                for u, v, w, cw in zip(
                    u_idx, v_idx, raw_sim_matrix[u_idx, v_idx], topological_affinity[u_idx, v_idx]
                )
            )

        logger.info(
            f"Clinical Graph constructed: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges (avg latency: {np.mean(time_to_process):.1f} ms/img)."
        )
        return G, image_embeddings, time_to_process


# STRUCTURED CLINICAL AUDIT PAYLOAD

@dataclass
class ClinicalRetrievalResult:
    """
    Structured payload containing Graph RAG retrieval results and audit metrics.

    Attributes:
        query: the submitted query.
        seed_nodes: top-ranked entry node IDs.
        retrieved_subgraph: seeds + ranked neighbours.
        smoothed_sims: 1-hop smoothed scores over all nodes (cosine scale; z-scale if the lexical channel fired).
        ppr_scores: Personalized PageRank mass per node.
        confidence_z_score: (top - mean) / std of `smoothed_sims`.
        is_low_confidence_warning: z below threshold (only when the threshold is attainable for this graph size).
        latency_ms: end-to-end retrieval latency.
        system_consensus: label frequency across retrieved nodes.
        lexical_weight_used: BM25 fusion weight actually applied (0.0 = dense only).
    """
    query: str
    seed_nodes: list[Any]
    retrieved_subgraph: list[Any]
    smoothed_sims: np.ndarray
    ppr_scores: dict[Any, float]
    confidence_z_score: float
    is_low_confidence_warning: bool
    latency_ms: float
    system_consensus: dict[str, int]
    lexical_weight_used: float = 0.0


# STATEFUL GRAPH RAG ENGINE (PRE-CACHED & VECTORIZED)

class UNClinicalGraphRAGEngine:
    """
    Cross-modal Graph RAG engine whose query side is a text.py `TextEmbedder`.

    Pre-computes and caches:
      - (optionally mean-centred) L2-normalised image embeddings on the embedder's device,
      - CSR row-stochastic smoothing matrix P = D^-1 A and its transpose for power-iteration PPR,
      - per-layer mean query vector (text-side centring) and per-layer token statistics in the embedder,
      - an optional BM25 index over node text labels.
    """

    def __init__(
        self,
        G: nx.Graph,
        embedder: Any,
        device: str | torch.device | None = None,
        fallback_embs: np.ndarray | None = None,
        confidence_z_threshold: float = 2.0,
        fit_queries: Sequence[str] | None = None,
        layers: Sequence[int] = (0, 1),
        layer_weights: Sequence[float] | None = None,
        query_center: bool = True,
        drop_stopwords: bool = True,
        lexical_weight: float = 0.3,
        center_images: bool | None = None,
        min_nodes_for_centering: int = 8,
        **kwargs: Any,
    ):
        """
        Args:
            G: knowledge graph (nodes carry 'embedding' and optionally 'text').
            embedder: a text.py `TextEmbedder` (provides model, tokenizer and token conditioning).
            device: compute device for scoring (default: the embedder model's device).
            fallback_embs: used when nodes lack an 'embedding' attribute.
            confidence_z_threshold: z-score under which a low-confidence warning is raised.
            fit_queries: optional unlabeled example queries of YOUR domain, used only for calibration
                (token statistics and the mean query vector). Built-in generic imaging phrases are always added.
            layers: hidden-state indices used for the query (0 = token embeddings, negatives allowed).
            layer_weights: optional weight per entry of `layers` (default: uniform).
            query_center: subtract the per-layer mean query vector (set False to A/B test).
            drop_stopwords: remove stopwords from the query before encoding.
            lexical_weight: BM25 fusion weight when >= 50% of nodes have text labels (0 = off).
            center_images: mean-centre image embeddings. None = auto (only if num_nodes >= min_nodes_for_centering).
            min_nodes_for_centering: see above.

        Raises:
            ValueError: empty graph, missing embedder, embedding-size mismatch, or unresolvable embeddings.
        """
        if G.number_of_nodes() == 0:
            raise ValueError("Graph G contains no nodes.")

        # Handle legacy method mapping if provided in kwargs
        method = kwargs.pop("method", None)
        if method == "fast_layer0_static":
            layers = (0,)
            layer_weights = (1.0,)
        elif method == "hybrid_blended_context":
            layers = (0, 1)
            layer_weights = (0.8, 0.2)

        # Handle legacy text_model + tokenizer wrapping
        if embedder is not None and not hasattr(embedder, "encode_raw"):
            tok = kwargs.pop("tokenizer", None) or getattr(embedder, "tokenizer", None)
            if tok is not None:
                from metalliger.utils.text import TextEmbedder
                embedder = TextEmbedder(model=embedder, tokenizer=tok)

        if embedder is None:
            text_model = kwargs.pop("text_model", None)
            tok = kwargs.pop("tokenizer", None)
            if text_model is not None and tok is not None:
                from metalliger.utils.text import TextEmbedder
                embedder = TextEmbedder(model=text_model, tokenizer=tok)
            else:
                raise ValueError("`embedder` (a text.py TextEmbedder) is required.")

        self.G = G
        self.embedder = embedder
        self.confidence_z_threshold = confidence_z_threshold
        self.query_center = query_center
        self.drop_stopwords = drop_stopwords
        self.lexical_weight = lexical_weight
        self.text_layers = tuple(layers)
        self._cache_key: tuple | None = None

        self.device = torch.device(embedder.model.device) if device is None else torch.device(device)

        # 1. Node-ID <-> contiguous index mappings
        self.node_list: list[Any] = list(G.nodes())
        self.num_nodes: int = len(self.node_list)
        self.node_to_idx: dict[Any, int] = {n: i for i, n in enumerate(self.node_list)}

        # 2. Image embeddings, calibrated ONCE
        raw_embs = self._extract_aligned_embeddings(fallback_embs)
        do_center = (self.num_nodes >= min_nodes_for_centering) if center_images is None else center_images
        if not do_center:
            logger.info(
                f"Image mean-centring skipped ({self.num_nodes} nodes < {min_nodes_for_centering}); "
                "the mean of so few images would erase their differences."
            )
        self.cal_img_np = self.calibrate_image_embeddings(raw_embs, center=do_center)
        self.cal_img_tensor = torch.as_tensor(self.cal_img_np, dtype=torch.float32, device=self.device)

        # 3. Query-side calibration through the embedder, ONCE
        self.emb_layer_mean: torch.Tensor | None = None
        self.layer_w: torch.Tensor | None = None
        self._init_embedder_path(fit_queries, layer_weights)

        # 4. Sparse graph operators ONCE
        self._build_sparse_graph_operators()

        # 5. Optional lexical channel over node text labels
        self.lex: _NodeLexicalIndex | None = None
        labels = [self.G.nodes[n].get("text") for n in self.node_list]
        labels = [str(t) if t else None for t in labels]
        if lexical_weight > 0 and sum(t is not None for t in labels) >= 0.5 * self.num_nodes:
            self.lex = _NodeLexicalIndex(labels)

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _extract_aligned_embeddings(self, fallback_embs: np.ndarray | None) -> np.ndarray:
        """Image embeddings aligned with self.node_list (node attributes first, then fallback_embs)."""
        if all("embedding" in self.G.nodes[n] for n in self.node_list):
            return np.vstack([self.G.nodes[n]["embedding"] for n in self.node_list]).astype(np.float32)
        if fallback_embs is not None:
            if fallback_embs.shape[0] < self.num_nodes:
                raise ValueError(
                    f"fallback_embs has {fallback_embs.shape[0]} rows, but Graph G has {self.num_nodes} nodes."
                )
            return np.asarray(fallback_embs[: self.num_nodes], dtype=np.float32)
        raise ValueError("No embeddings found in G node attributes or fallback_embs.")

    @staticmethod
    def calibrate_image_embeddings(
        img_embs: np.ndarray, center: bool = True
    ) -> np.ndarray:
        """Mean-centres (optional) and L2-normalises image embeddings. Returns [N, D]."""
        if img_embs.ndim != 2 or img_embs.shape[0] == 0:
            raise ValueError("img_embs must be a non-empty 2D array.")
        img_embs = np.asarray(img_embs, dtype=np.float32)
        img_mean = img_embs.mean(axis=0, keepdims=True) if center else np.zeros((1, img_embs.shape[1]), np.float32)
        centered = img_embs - img_mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.clip(norms, 1e-9, None)

    def _build_sparse_graph_operators(self) -> None:
        """CSR smoothing matrix P = D^-1 A, its transpose for PPR, and dangling/has-neighbour masks."""
        if self.G.number_of_edges() == 0:
            empty = sp.csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            self.smoothing_matrix = empty
            self.ppr_transition_T = empty
            self.dangling_mask = np.ones(self.num_nodes, dtype=bool)
            self.has_neighbors_mask = np.zeros(self.num_nodes, dtype=bool)
            return

        adj = nx.to_scipy_sparse_array(
            self.G, nodelist=self.node_list, weight="weight", format="csr", dtype=np.float32
        )
        row_sums = np.asarray(adj.sum(axis=1)).reshape(-1)
        self.has_neighbors_mask = row_sums > 0.0
        self.dangling_mask = ~self.has_neighbors_mask

        inv_deg = np.zeros_like(row_sums, dtype=np.float32)
        inv_deg[self.has_neighbors_mask] = 1.0 / row_sums[self.has_neighbors_mask]

        P = sp.diags(inv_deg) @ adj
        self.smoothing_matrix = P.tocsr()
        self.ppr_transition_T = P.T.tocsr()

    def get_node_image_info(self, node_id: Any) -> tuple[str, str]:
        """Resolves (image name, label) for a node."""
        node_data = self.G.nodes[node_id]
        label = node_data.get("text")
        if label is None:
            label = node_data.get("body_system", "Unknown")

        raw_img = node_data.get("image", node_data.get("path", None))
        if isinstance(raw_img, (list, tuple)) and len(raw_img) > 0:
            img_name = str(raw_img[0])
        elif isinstance(raw_img, str):
            img_name = raw_img
        else:
            img_name = f"node_{node_id}"
        return img_name, str(label)

    # ------------------------------------------------------------------ #
    # Query encoding through the text.py TextEmbedder
    # ------------------------------------------------------------------ #
    def _prep_query(self, query: str) -> str:
        """NFKC-normalise and (optionally) drop stopwords, keeping the query if nothing else is left."""
        norm = " ".join(unicodedata.normalize("NFKC", str(query)).split())
        if not norm:
            raise ValueError("Query must not be empty.")
        if self.drop_stopwords:
            words = norm.split()
            kept = [w for w in words if w.lower().strip(string.punctuation) not in _STOPWORDS]
            norm = " ".join(kept or words)
        return norm

    @torch.inference_mode()
    def _embedder_raw(self, texts: Sequence[str]) -> torch.Tensor:
        """[n, L, D] mean-pooled states (first half of `encode_raw`) for `self.text_layers`."""
        cfg = self.embedder.config
        old = cfg.layers
        cfg.layers = self.text_layers
        try:
            raw = self.embedder.encode_raw(list(texts), "query")
        finally:
            cfg.layers = old
        d = self.embedder.model.config.hidden_size
        return raw[:, :, :d].float().to(self.device)   # image vectors are mean-pooled: drop the max half

    @torch.inference_mode()
    def _init_embedder_path(self, fit_queries: Sequence[str] | None, layer_weights: Sequence[float] | None) -> None:
        """Validates dims/layers, calibrates token stats for layers the embedder has not seen, and estimates
        the per-layer mean query vector (text-side centring)."""
        emb = self.embedder
        d_text, d_img = emb.model.config.hidden_size, self.cal_img_tensor.shape[-1]
        if d_text != d_img:
            raise ValueError(f"Embedder hidden size {d_text} != image embedding size {d_img}.")
        self.text_layers = tuple(emb._norm_layers(self.text_layers))

        if layer_weights is None:
            w = torch.ones(len(self.text_layers))
        else:
            if len(layer_weights) != len(self.text_layers):
                raise ValueError("`layer_weights` must have one entry per layer in `layers`.")
            w = torch.as_tensor(list(layer_weights), dtype=torch.float32)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError("`layer_weights` must be non-negative with a positive sum.")
        self.layer_w = (w / w.sum()).to(self.device)

        cal: list[str] = []
        for t in dict.fromkeys(list(fit_queries or []) + list(_DEFAULT_CAL_QUERIES)):
            try:
                cal.append(self._prep_query(t))
            except ValueError:
                continue
        cal = list(dict.fromkeys(cal))

        cfg = emb.config
        missing = [l for l in self.text_layers if l not in emb.tok_stats]
        if missing and (cfg.token_zscore or cfg.mask_outlier_tokens):
            old = cfg.layers
            cfg.layers = tuple(missing)
            try:
                emb.calibrate(cal)
            finally:
                cfg.layers = old
        if self.query_center:
            self.emb_layer_mean = self._embedder_raw(cal).mean(dim=0)   # [L, D]

    @torch.inference_mode()
    def score_with_embedder(self, query: str) -> np.ndarray:
        """Cosine similarity of the embedder-encoded query (selected layers, weighted) to every image."""
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        z = self._embedder_raw([self._prep_query(query)])[0]        # [L, D]
        if self.emb_layer_mean is not None:
            z = z - self.emb_layer_mean
        z = F.normalize(z, p=2, dim=-1)
        q_vec = F.normalize((z * self.layer_w.unsqueeze(-1)).sum(dim=0, keepdim=True), p=2, dim=-1)
        scores = q_vec.to(self.cal_img_tensor.dtype) @ self.cal_img_tensor.T
        return scores.squeeze(0).float().cpu().numpy()

    def _fuse_lexical(self, query: str, dense: np.ndarray, w: float) -> tuple[np.ndarray, float]:
        """(1-w) * z(dense) + w * z(BM25 over node labels). No-op when there is no lexical signal."""
        if self.lex is None or w <= 0.0:
            return dense, 0.0
        lex = self.lex.score(query)
        if lex.max() <= 0.0:
            return dense, 0.0
        return (1.0 - w) * _zs(dense) + w * _zs(lex), w

    # ------------------------------------------------------------------ #
    # Graph retrieval
    # ------------------------------------------------------------------ #
    def _fast_personalized_pagerank(
        self,
        personalization_vec: np.ndarray,
        alpha: float = 0.85,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> dict[Any, float]:
        """Power-iteration Personalized PageRank on the cached CSR transition matrix."""
        p = personalization_vec / np.clip(personalization_vec.sum(), 1e-12, None)
        x = p.copy()
        for _ in range(max_iter):
            x_last = x
            dangling_sum = x[self.dangling_mask].sum()
            x = alpha * (self.ppr_transition_T @ x + dangling_sum * p) + (1.0 - alpha) * p
            if np.abs(x - x_last).sum() < self.num_nodes * tol:
                break
        return {self.node_list[i]: float(x[i]) for i in range(self.num_nodes)}

    def query(
        self,
        text_query: str,
        top_k_seeds: int = 1,
        k_neighbors: int = K_NEIGHBORS,
        smoothing_lambda: float = 0.40,
        verbose: bool = True,
    ) -> ClinicalRetrievalResult:
        """
        Full retrieval: embedder scoring -> (BM25 fusion) -> 1-hop smoothing -> seeds + confidence
        -> PPR -> ranked neighbours.

        Args:
            text_query: query string.
            top_k_seeds: number of entry seeds.
            k_neighbors: max neighbours returned.
            smoothing_lambda: weight of neighbourhood smoothing (0 = off).
            verbose: print the audit table.
        """
        t0 = time.perf_counter()

        # Step A: cross-modal similarity (+ optional lexical fusion over node labels)
        raw_sims = self.score_with_embedder(text_query)
        raw_sims, lex_used = self._fuse_lexical(text_query, raw_sims, self.lexical_weight)

        # Step B: 1-hop graph signal smoothing
        smoothed_sims = np.copy(raw_sims)
        if self.G.number_of_edges() > 0 and smoothing_lambda > 0.0:
            nbr_means = self.smoothing_matrix @ raw_sims
            mask = self.has_neighbors_mask
            smoothed_sims[mask] = (1.0 - smoothing_lambda) * raw_sims[mask] + smoothing_lambda * nbr_means[mask]

        # Step C: seeds + confidence. A z-score over N nodes can never exceed (N-1)/sqrt(N), so for small
        # graphs the threshold is unreachable and the warning would always fire; it is suppressed there.
        sorted_indices = np.argsort(smoothed_sims)[::-1]
        seed_nodes = [self.node_list[i] for i in sorted_indices[:top_k_seeds]]

        sim_mean = float(smoothed_sims.mean())
        sim_std = max(float(smoothed_sims.std()), 1e-6)
        top_score = float(smoothed_sims[sorted_indices[0]])
        confidence_z = (top_score - sim_mean) / sim_std
        z_attainable = (self.num_nodes - 1) / np.sqrt(self.num_nodes) > self.confidence_z_threshold
        low_conf_warning = bool(z_attainable and confidence_z < self.confidence_z_threshold)

        # Step D: Personalized PageRank rooted on the similarity distribution
        temp_scores = (smoothed_sims - top_score) / max(sim_std, 1e-4)
        exp_scores = np.exp(np.clip(temp_scores, -50.0, 0.0))
        ppr_scores = self._fast_personalized_pagerank(exp_scores, alpha=0.85)

        candidate_neighbors = set()
        for seed in seed_nodes:
            candidate_neighbors.update(self.G.neighbors(seed))
        candidate_neighbors -= set(seed_nodes)

        ranked_neighbors = sorted(
            candidate_neighbors,
            key=lambda n: (
                max(self.G[n][s].get("weight", 0.85) for s in seed_nodes if self.G.has_edge(n, s))
                * ppr_scores.get(n, 0.0)
            ),
            reverse=True,
        )

        retrieved_subgraph = seed_nodes + ranked_neighbors[:k_neighbors]
        consensus = dict(Counter(self.get_node_image_info(n)[1] for n in retrieved_subgraph))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if verbose:
            print("=" * 88)
            print(
                f"UN CLINICAL GRAPH RAG | LAYERS: {list(self.text_layers)} | QUERY: '{text_query}' | "
                f"Latency: {latency_ms:.2f} ms | lexical w={lex_used}"
            )
            print(
                f"Scores -> Max: {top_score:.4f} | Min: {smoothed_sims.min():.4f} | "
                f"Mean: {sim_mean:.4f} | Std: {sim_std:.4f} | Confidence Z: {confidence_z:.2f}σ"
            )
            if low_conf_warning:
                print(
                    f"⚠️ [CLINICAL WARNING] Low retrieval margin (Z={confidence_z:.2f}σ < "
                    f"{self.confidence_z_threshold}σ). Verify query specificity."
                )
            elif not z_attainable:
                print(f"(Z-score warning disabled: {self.num_nodes} nodes are too few for a {self.confidence_z_threshold}σ margin.)")
            print("-" * 88)
            for node_id in retrieved_subgraph:
                node_idx = self.node_to_idx[node_id]
                role = "[SEED]" if node_id in seed_nodes else "[NBR] "
                img_name, body_sys = self.get_node_image_info(node_id)
                print(
                    f"  {role} Node {str(node_id):<4} | System: {body_sys:<18} | "
                    f"Sim: {smoothed_sims[node_idx]:+.4f} | PPR: {ppr_scores[node_id]:.5f} | {img_name}"
                )
            print("Subgraph Clinical Consensus:", consensus)

        return ClinicalRetrievalResult(
            query=text_query,
            seed_nodes=seed_nodes,
            retrieved_subgraph=retrieved_subgraph,
            smoothed_sims=smoothed_sims,
            ppr_scores=ppr_scores,
            confidence_z_score=confidence_z,
            is_low_confidence_warning=low_conf_warning,
            latency_ms=latency_ms,
            system_consensus=consensus,
            lexical_weight_used=lex_used,
        )


class GraphRAG:
    """
    High-level Graph RAG retrieval service. Queries are encoded exclusively by a text.py
    `TextEmbedder`. The engine is built lazily on the first call and rebuilt automatically when the
    graph, embedder, layers or engine options change.
    """

    def __init__(self):
        self.last_result: ClinicalRetrievalResult | None = None

    @staticmethod
    def reset() -> None:
        """Drops the cached engine (e.g. after editing node attributes in place)."""
        global _CACHED_ENGINE
        _CACHED_ENGINE = None

    def query_graph_rag_pure(
        self,
        text_query: str,
        G: nx.Graph,
        embedder: Any = None,
        img_embeddings: np.ndarray | None = None,
        top_k_seeds: int = 1,
        k_neighbors: int = K_NEIGHBORS,
        smoothing_lambda: float = 0.40,
        verbose: bool = True,
        layers: Sequence[int] = (0, 1),
        layer_weights: Sequence[float] | None = None,
        fit_queries: Sequence[str] | None = None,
        force_rebuild: bool = False,
        **engine_kwargs: Any,
    ) -> tuple[list[Any], list[Any], np.ndarray]:
        """
        Entry point for Graph RAG retrieval.

        Args:
            text_query: input text prompt or clinical question.
            G: knowledge graph containing image nodes.
            embedder: text.py `TextEmbedder` (or `(text_model, tokenizer)`).
            img_embeddings: fallback image embeddings if nodes lack an 'embedding' attribute.
            top_k_seeds: number of seed nodes.
            k_neighbors: number of neighbour nodes.
            smoothing_lambda: neighbourhood smoothing factor.
            verbose: print the retrieval audit.
            layers: hidden-state indices used for the query (0 = token embeddings).
            layer_weights: optional weight per layer (default uniform).
            fit_queries: optional unlabeled domain queries for calibration (engine build only).
            force_rebuild: rebuild the cached engine.
            **engine_kwargs: forwarded to UNClinicalGraphRAGEngine (query_center, drop_stopwords,
                lexical_weight, center_images, method, text_model, tokenizer, ...).

        Returns:
            (seed_nodes, retrieved_subgraph, smoothed_sims). The full payload is kept in `self.last_result`.
        """
        global _CACHED_ENGINE

        # Backwards compatibility: handle method mapping
        method = engine_kwargs.pop("method", None)
        if method == "fast_layer0_static":
            layers = (0,)
            layer_weights = (1.0,)
        elif method == "hybrid_blended_context":
            layers = (0, 1)
            layer_weights = (0.8, 0.2)

        # Backwards compatibility: handle legacy positional or keyword text_model + tokenizer
        if embedder is None:
            text_model = engine_kwargs.pop("text_model", None)
            tokenizer = engine_kwargs.pop("tokenizer", None)
            if text_model is not None and tokenizer is not None:
                from metalliger.utils.text import TextEmbedder
                embedder = TextEmbedder(model=text_model, tokenizer=tokenizer)
            else:
                raise ValueError("`embedder` (a text.py TextEmbedder) or `(text_model, tokenizer)` is required.")
        elif not hasattr(embedder, "encode_raw"):
            if hasattr(img_embeddings, "encode") or hasattr(img_embeddings, "decode"):
                from metalliger.utils.text import TextEmbedder
                real_tokenizer = img_embeddings
                embedder = TextEmbedder(model=embedder, tokenizer=real_tokenizer)
                img_embeddings = engine_kwargs.pop("img_embeddings", None)
                if img_embeddings is None and not isinstance(top_k_seeds, int):
                    img_embeddings = top_k_seeds
                    top_k_seeds = 1

        key = (id(G), G.number_of_nodes(), G.number_of_edges(), id(embedder), tuple(layers),
               tuple(layer_weights) if layer_weights is not None else None,
               tuple(sorted((k, repr(v)) for k, v in engine_kwargs.items())), tuple(fit_queries or ()))
        if force_rebuild or _CACHED_ENGINE is None or _CACHED_ENGINE._cache_key != key:
            _CACHED_ENGINE = UNClinicalGraphRAGEngine(
                G=G,
                embedder=embedder,
                fallback_embs=img_embeddings,
                fit_queries=fit_queries,
                layers=tuple(layers),
                layer_weights=layer_weights,
                **engine_kwargs,
            )
            _CACHED_ENGINE._cache_key = key

        res = _CACHED_ENGINE.query(
            text_query=text_query,
            top_k_seeds=top_k_seeds,
            k_neighbors=k_neighbors,
            smoothing_lambda=smoothing_lambda,
            verbose=verbose,
        )
        self.last_result = res
        return res.seed_nodes, res.retrieved_subgraph, res.smoothed_sims
