"""
Clinical Text Embedding and Graph RAG Module for UN Hospital Deployment.

This module mirrors the Vision Graph RAG architecture (`vison_2.py`) for the
standalone `Qwen3VLTextModel` (Anatomy-Dominant 2048-D embedding tower). It includes:
  - UTF-8 clinical text normalization and SHA-256 EHR audit hashing.
  - Exact core-sliced micro-batched text embedding extraction with 3D M-RoPE.
  - Anisotropy correction (mean-centering) and CSLS hubness suppression.
  - Vectorized Mutual k-NN similarity graph construction (with optional vision fusion).
  - Precomputed CSR sparse graph operators (1-hop smoothing & Personalized PageRank).
  - Multi-tier clinical query scoring (Layer-0 Static, Layer-0/1 Vision Boundary, and DeepStack [5, 11, 17]).
"""

import hashlib
import logging
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
from transformers import AutoTokenizer, Qwen3VLTextModel

# Configure logger for UN Clinical Text Graph RAG operations
logger = logging.getLogger("UNClinicalTextGraphRAG")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Default number of neighboring nodes retrieved during Graph RAG traversal
K_NEIGHBORS = 10

# Special token IDs marking the vision boundaries in Qwen3-VL chat templates
VISION_START_ID = 151652
VISION_END_ID = 151653
DEEPSTACK_LAYERS = (5, 11, 17)

# Global engine cache reference for lazy initialization
_CACHED_TEXT_ENGINE: Any = None


def get_default_device() -> torch.device:
    """Selects CUDA (NVIDIA Server), then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextFunc:
    """
    Clinical Text Embedding & Graph Construction Utilities.

    Mirrors `func` from `vison_2.py` for clinical text records, radiology reports,
    ICD/SNOMED descriptions, and VQA entries using `Qwen3VLTextModel`.
    """

    def __init__(self, device: str | torch.device | None = None):
        """
        Initializes the clinical text helper utility class.

        Args:
            device (str | torch.device | None, optional): Target accelerator ('cuda', 'mps', 'cpu').
                Defaults to automatic hardware detection.
        """
        self.device = torch.device(device) if device else get_default_device()

    @staticmethod
    def _clean_and_hash_text(text_input: str) -> tuple[str, str]:
        """
        Normalizes international clinical text and generates a SHA-256 audit hash.

        Applies Unicode NFKC normalization (handling international accents/symbols
        across UN member-state clinical records), collapses redundant whitespace, and
        computes a 16-character SHA-256 digest for EHR deduplication and audit trails.

        Args:
            text_input (str): Raw clinical text, report, or anatomical label.

        Returns:
            tuple[str, str]:
                - normalized_text: Cleaned UTF-8 clinical string.
                - text_hash: First 16 hex characters of the SHA-256 digest.

        Raises:
            TypeError: If text_input is not a string.
            ValueError: If text_input is empty after normalization.
        """
        if not isinstance(text_input, str):
            raise TypeError(f"Expected string input, got {type(text_input)}")

        normalized_text = unicodedata.normalize("NFKC", text_input)
        normalized_text = " ".join(normalized_text.strip().split())

        if not normalized_text:
            raise ValueError("Clinical text input is empty after normalization.")

        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]
        return normalized_text, text_hash

    @torch.inference_mode()
    def encode_texts_batch(
        self,
        tokenizer: AutoTokenizer,
        text_model: Qwen3VLTextModel,
        texts: Sequence[str],
        layer0_weight: float = 0.80,
        use_deepstack: bool = False,
    ) -> np.ndarray:
        """
        Encodes a batch of clinical texts using Qwen3-VL Vision-Boundary Alignment.

        Wraps each text in `<|im_start|><|vision_start|> {text} <|vision_end|><|im_end|>`.
        Because Qwen3VLTextModel uses causal self-attention, right-padding after `<|im_end|>`
        does not alter the hidden states of the inner query tokens. A per-sample core token
        mask slices strictly the inner medical tokens, guaranteeing 100% mathematical
        equivalence to single-sample encoding at 10x-20x higher throughput.

        Args:
            tokenizer (AutoTokenizer): Qwen3-VL tokenizer.
            text_model (Qwen3VLTextModel): Standalone Qwen3-VL text backbone.
            texts (Sequence[str]): List of cleaned clinical strings to encode.
            layer0_weight (float, optional): Blending weight for Layer 0 (`embed_tokens`)
                vs. Layer 1 contextual hidden states. When 1.0, skips the 28-layer
                Transformer pass completely. Defaults to 0.80.
            use_deepstack (bool, optional): If True, pools intermediate DeepStack visual-injection
                layers (5, 11, 17) alongside Layer 0. Defaults to False.

        Returns:
            np.ndarray: L2-normalized float32 embeddings of shape [batch_size, 2048] on CPU.
        """
        if not texts:
            raise ValueError("`texts` batch cannot be empty.")
        if not 0.0 <= layer0_weight <= 1.0:
            raise ValueError("`layer0_weight` must be between 0.0 and 1.0.")

        sprefix_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        nprefix_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        batch_sequences = []
        core_spans = []
        start_offset = len(sprefix_ids) + 1  # Position immediately after VISION_START_ID

        for txt in texts:
            core_ids = tokenizer.encode(txt, add_special_tokens=False)
            if not core_ids:
                core_ids = tokenizer.encode(txt, add_special_tokens=True)

            seq = sprefix_ids + [VISION_START_ID] + core_ids + [VISION_END_ID] + nprefix_ids
            batch_sequences.append(seq)
            core_spans.append((start_offset, start_offset + len(core_ids)))

        batch_size = len(batch_sequences)
        max_len = max(len(s) for s in batch_sequences)

        # Construct right-padded input_ids, attention_mask, and exact inner core_mask
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        core_mask = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=self.device)

        for i, (seq, (c_start, c_end)) in enumerate(zip(batch_sequences, core_spans)):
            seq_len = len(seq)
            input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attn_mask[i, :seq_len] = 1
            core_mask[i, c_start:c_end, 0] = 1.0

        core_counts = core_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]

        # 1. Extract Layer 0 (static input token space aligned with Vision Patch Merger)
        layer0 = text_model.embed_tokens(input_ids).to(torch.float32)  # [B, max_len, 2048]
        inner_l0 = (layer0 * core_mask).sum(dim=1) / core_counts
        e0 = F.normalize(inner_l0, p=2, dim=-1)

        need_transformer = (layer0_weight < 1.0 - 1e-6) or use_deepstack
        if not need_transformer:
            return e0.cpu().numpy()

        # 2. Compute 3D M-RoPE position IDs: [3, B, max_len]
        pos_1d = (attn_mask.long().cumsum(-1) - 1).masked_fill(attn_mask == 0, 1)
        pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)

        outputs = text_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_3d,
            output_hidden_states=True,
            return_dict=True,
        )

        if use_deepstack:
            # Pool across Qwen3-VL's DeepStack injection layers [5, 11, 17]
            ds_pools = []
            for l_idx in DEEPSTACK_LAYERS:
                if l_idx + 1 < len(outputs.hidden_states):
                    h = outputs.hidden_states[l_idx + 1].to(torch.float32)
                    p = F.normalize((h * core_mask).sum(dim=1) / core_counts, p=2, dim=-1)
                    ds_pools.append(p)
            e_context = F.normalize(torch.stack(ds_pools, dim=0).mean(dim=0), p=2, dim=-1)
        else:
            layer1 = outputs.hidden_states[1].to(torch.float32)
            inner_l1 = (layer1 * core_mask).sum(dim=1) / core_counts
            e_context = F.normalize(inner_l1, p=2, dim=-1)

        blended = F.normalize(
            layer0_weight * e0 + (1.0 - layer0_weight) * e_context, p=2, dim=-1
        )
        result = blended.cpu().numpy()

        del input_ids, attn_mask, core_mask, layer0, outputs, blended
        return result

    def similarity_graph(
        self,
        texts: Sequence[str],
        tokenizer: AutoTokenizer,
        text_model: Qwen3VLTextModel,
        labels: Sequence[str | None] | None = None,
        images: Sequence[str | None] | None = None,
        vision_embeddings: np.ndarray | None = None,
        similarity_threshold: float = 0.85,
        batch_size: int = 32,
        layer0_weight: float = 0.80,
        max_neighbors_k: int = 12,
        mutual_knn: bool = True,
        calibrate_hubness: bool = True,
        hybrid_vision_alpha: float = 0.35,
    ) -> tuple[nx.Graph, np.ndarray, list[float]]:
        """
        Constructs an enterprise/clinical-grade Anatomy-Dominant similarity graph from text records.

        Mirrors `func.similarity_graph` from `vison_2.py`:
          1. Validates, normalizes, and hashes clinical text inputs.
          2. Micro-batched GPU/MPS text embedding extraction (`encode_texts_batch`).
          3. Mean-centering anisotropy calibration and CSLS hubness suppression.
          4. Optional hybrid fusion with `vision_embeddings` (combining Anatomy-Dominant text
             with Modality-Dominant vision).
          5. Vectorized Mutual k-NN edge construction (no O(N^2) Python loops).

        Args:
            texts (Sequence[str]): Clinical text records, reports, or VQA entries.
            tokenizer (AutoTokenizer): Qwen3-VL tokenizer.
            text_model (Qwen3VLTextModel): Qwen3-VL text embedding backbone.
            labels (Sequence[str | None] | None, optional): Ground-truth anatomical or
                composite `body_system` labels per record. Defaults to None.
            images (Sequence[str | None] | None, optional): Optional associated image file paths
                stored on nodes for multimodal cross-reference. Defaults to None.
            vision_embeddings (np.ndarray | None, optional): Precomputed [N, 2048] image embeddings
                for hybrid edge fusion. Defaults to None.
            similarity_threshold (float, optional): Minimum cosine similarity in [0, 1]. Defaults to 0.85.
            batch_size (int, optional): Micro-batch size for text encoding. Defaults to 32.
            layer0_weight (float, optional): Layer 0 vs. Layer 1 fusion weight. Defaults to 0.80.
            max_neighbors_k (int, optional): Max top-k candidates per node. Defaults to 12.
            mutual_knn (bool, optional): Require bidirectional top-k agreement. Defaults to True.
            calibrate_hubness (bool, optional): Apply mean-centering + CSLS scaling. Defaults to True.
            hybrid_vision_alpha (float, optional): Weight of visual similarity when `vision_embeddings`
                is supplied. Defaults to 0.35.

        Returns:
            tuple[nx.Graph, np.ndarray, list[float]]:
                - G: Undirected NetworkX Graph with node attributes ('clinical_text', 'text',
                     'body_system', 'image', 'sha256', 'embedding').
                - text_embeddings: L2-normalized NumPy array of shape [N, 2048].
                - time_to_process: Per-record encoding latency in milliseconds.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError("`texts` must be a list or tuple of clinical strings.")
        if labels is not None and len(labels) != len(texts):
            raise ValueError("`labels` and `texts` must have the exact same length.")
        if images is not None and len(images) != len(texts):
            raise ValueError("`images` and `texts` must have the exact same length.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("`similarity_threshold` must be between 0.0 and 1.0.")
        if not texts:
            raise ValueError("The `texts` list is empty.")

        valid_records = []
        for idx, raw_txt in enumerate(texts):
            if raw_txt is None or not str(raw_txt).strip():
                logger.warning(f"Skipping index {idx}: clinical text is empty or None.")
                continue
            clean_txt, sha_hash = self._clean_and_hash_text(str(raw_txt))
            lbl = labels[idx] if labels is not None else clean_txt
            img_ref = images[idx] if images is not None else f"record_{idx}"
            valid_records.append((idx, clean_txt, lbl, img_ref, sha_hash))

        if not valid_records:
            raise ValueError("No valid clinical text records were found to process.")

        embeddings_batches = []
        time_to_process: list[float] = []
        node_metadata = []

        total_valid = len(valid_records)
        logger.info(
            f"Encoding {total_valid} clinical text records (batch_size={batch_size}) on {self.device}..."
        )

        for b_start in range(0, total_valid, batch_size):
            batch_slice = valid_records[b_start : b_start + batch_size]
            batch_strings = [rec[1] for rec in batch_slice]

            t0 = time.perf_counter()
            batch_embs = self.encode_texts_batch(
                tokenizer=tokenizer,
                text_model=text_model,
                texts=batch_strings,
                layer0_weight=layer0_weight,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_rec_ms = round(elapsed_ms / len(batch_strings), 2)

            embeddings_batches.append(batch_embs)
            time_to_process.extend([per_rec_ms] * len(batch_strings))

            for orig_idx, clean_txt, lbl, img_ref, sha_hash in batch_slice:
                node_metadata.append({
                    "orig_idx": orig_idx,
                    "clinical_text": clean_txt,
                    "label": str(lbl),
                    "image_ref": img_ref,
                    "sha256": sha_hash,
                })

        if self.device.type == "mps":
            torch.mps.empty_cache()

        text_embeddings = np.vstack(embeddings_batches).astype(np.float32)
        num_nodes = text_embeddings.shape[0]

        # Compute raw cosine similarity matrix
        raw_sim_matrix = np.dot(text_embeddings, text_embeddings.T)

        if calibrate_hubness and num_nodes > 10:
            mean_vec = text_embeddings.mean(axis=0, keepdims=True)
            centered = text_embeddings - mean_vec
            cal_embs = centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(min=1e-9)
            cal_sim = np.dot(cal_embs, cal_embs.T)

            k_csls = min(10, num_nodes - 1)
            knn_mean_sim = np.partition(cal_sim, -(k_csls + 1), axis=1)[:, -(k_csls + 1) : -1].mean(axis=1)
            csls_matrix = cal_sim - 0.5 * (knn_mean_sim[:, None] + knn_mean_sim[None, :])
            topological_affinity = 0.5 * raw_sim_matrix + 0.5 * csls_matrix
        else:
            topological_affinity = raw_sim_matrix.copy()

        # Optional: Fuse Modality-Dominant Vision Embeddings if provided
        if vision_embeddings is not None and vision_embeddings.shape[0] == num_nodes:
            v_mean = vision_embeddings.mean(axis=0, keepdims=True)
            v_cal = (vision_embeddings - v_mean) / np.linalg.norm(
                vision_embeddings - v_mean, axis=1, keepdims=True
            ).clip(min=1e-9)
            vis_sim = np.dot(v_cal, v_cal.T)
            topological_affinity = (
                (1.0 - hybrid_vision_alpha) * topological_affinity
                + (hybrid_vision_alpha * vis_sim)
            )

        np.fill_diagonal(topological_affinity, -np.inf)
        np.fill_diagonal(raw_sim_matrix, 0.0)

        # Build NetworkX Graph
        G = nx.Graph()
        for idx, meta in enumerate(node_metadata):
            G.add_node(
                idx,
                clinical_text=meta["clinical_text"],
                text=meta["label"],
                body_system=meta["label"],
                image=meta["image_ref"],
                sha256=meta["sha256"],
                embedding=text_embeddings[idx],
            )

        if num_nodes > 1:
            k_eff = min(max_neighbors_k, num_nodes - 1)
            top_k_indices = np.argpartition(topological_affinity, -k_eff, axis=1)[:, -k_eff:]

            knn_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
            row_idx = np.arange(num_nodes)[:, None]
            knn_mask[row_idx, top_k_indices] = True

            adj_mask = (knn_mask & knn_mask.T) if mutual_knn else (knn_mask | knn_mask.T)
            threshold_mask = raw_sim_matrix >= similarity_threshold
            final_edge_mask = np.triu(adj_mask & threshold_mask, k=1)

            u_indices, v_indices = np.nonzero(final_edge_mask)
            edge_weights = raw_sim_matrix[u_indices, v_indices]
            csls_weights = topological_affinity[u_indices, v_indices]

            G.add_edges_from(
                (int(u), int(v), {"weight": float(w), "calibrated_affinity": float(cw)})
                for u, v, w, cw in zip(u_indices, v_indices, edge_weights, csls_weights)
            )

        logger.info(
            f"Clinical Text Graph constructed: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges (avg latency: {np.mean(time_to_process):.2f} ms/record)."
        )
        return G, text_embeddings, time_to_process


@dataclass
class ClinicalTextRetrievalResult:
    """Structured audit payload for Clinical Text Graph RAG queries."""

    query: str
    method: str
    seed_nodes: list[Any]
    retrieved_subgraph: list[Any]
    smoothed_sims: np.ndarray
    ppr_scores: dict[Any, float]
    confidence_z_score: float
    is_low_confidence_warning: bool
    latency_ms: float
    system_consensus: dict[str, int]


class UNClinicalTextGraphRAGEngine:
    """
    Stateful Hospital Text Graph RAG Engine.

    Mirrors `UNClinicalGraphRAGEngine` for querying either a Pure Text Graph (`G_text`)
    or a Hybrid Clinical Graph using `Qwen3VLTextModel`.
    """

    def __init__(
        self,
        G: nx.Graph,
        text_model: Qwen3VLTextModel,
        tokenizer: AutoTokenizer,
        device: str | torch.device | None = None,
        image_dir: str = "./medmax_data/images_/",
        fallback_embs: np.ndarray | None = None,
        confidence_z_threshold: float = 2.0,
    ):
        if G.number_of_nodes() == 0:
            raise ValueError("Graph G contains no nodes.")

        self.G = G
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.confidence_z_threshold = confidence_z_threshold
        self.device = torch.device(device) if device else next(text_model.parameters()).device

        self.node_list: list[Any] = list(G.nodes())
        self.num_nodes: int = len(self.node_list)
        self.node_to_idx: dict[Any, int] = {
            node_id: idx for idx, node_id in enumerate(self.node_list)
        }

        raw_embs = self._extract_aligned_embeddings(fallback_embs)
        self.cal_embs_np, self.embs_mean_np = self.calibrate_embeddings(raw_embs)
        self.cal_embs_tensor = torch.as_tensor(
            self.cal_embs_np, dtype=torch.float32, device=self.device
        )

        self.sprefix_ids: list[int] = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        self.nprefix_ids: list[int] = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        self._build_sparse_graph_operators()

        self.methods_registry = {
            "1_layer0_boundary_pro": self.score_medmax_chat_aligned,
            "balanced_vision_boundary": self.score_medmax_chat_aligned,
            "fast_layer0_static": lambda q: self.score_medmax_chat_aligned(q, layer0_weight=1.0),
            "hybrid_blended_context": lambda q: self.score_medmax_chat_aligned(q, layer0_weight=0.80),
            "deepstack_clinical_context": lambda q: self.score_medmax_chat_aligned(
                q, layer0_weight=0.50, use_deepstack=True
            ),
        }

    def _extract_aligned_embeddings(self, fallback_embs: np.ndarray | None) -> np.ndarray:
        first_node = self.node_list[0]
        if "embedding" in self.G.nodes[first_node]:
            return np.vstack([self.G.nodes[n]["embedding"] for n in self.node_list]).astype(np.float32)
        if fallback_embs is not None:
            if fallback_embs.shape[0] < self.num_nodes:
                raise ValueError(
                    f"fallback_embs has {fallback_embs.shape[0]} rows, "
                    f"but Graph G has {self.num_nodes} nodes."
                )
            return np.asarray(fallback_embs[: self.num_nodes], dtype=np.float32)
        raise ValueError("No embeddings found in G node attributes or fallback_embs.")

    @staticmethod
    def calibrate_embeddings(embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if embs.ndim != 2 or embs.shape[0] == 0:
            raise ValueError("embs must be a non-empty 2D array.")
        embs = np.asarray(embs, dtype=np.float32)
        mean_vec = embs.mean(axis=0, keepdims=True)
        centered = embs - mean_vec
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return centered / np.clip(norms, 1e-9, None), mean_vec

    def _build_sparse_graph_operators(self) -> None:
        if self.G.number_of_edges() == 0:
            self.smoothing_matrix = sp.csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            self.ppr_transition_T = sp.csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
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

    def get_node_info(self, node_id: Any) -> tuple[str, str, str]:
        """Returns (image_or_record_name, clinical_text_snippet, body_system_label)."""
        node_data = self.G.nodes[node_id]
        label = node_data.get("body_system") or node_data.get("text", "Unknown")
        clinical_txt = node_data.get("clinical_text", str(label))
        raw_img = node_data.get("image", f"record_{node_id}")
        img_str = str(raw_img[0]) if isinstance(raw_img, (list, tuple)) and raw_img else str(raw_img)
        return img_str, str(clinical_txt), str(label)

    @torch.inference_mode()
    def score_medmax_chat_aligned(
        self,
        query: str,
        layer0_weight: float = 0.80,
        use_deepstack: bool = False,
    ) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
        if not 0.0 <= layer0_weight <= 1.0:
            raise ValueError("layer0_weight must be between 0.0 and 1.0.")

        core_ids = self.tokenizer.encode(query.strip(), add_special_tokens=False)
        if not core_ids:
            core_ids = self.tokenizer.encode(query.strip(), add_special_tokens=True)

        full_ids = (
            self.sprefix_ids
            + [VISION_START_ID]
            + core_ids
            + [VISION_END_ID]
            + self.nprefix_ids
        )
        start_idx = len(self.sprefix_ids) + 1
        end_idx = start_idx + len(core_ids)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)

        layer0 = self.text_model.embed_tokens(input_ids).to(torch.float32)
        e0 = F.normalize(layer0[:, start_idx:end_idx, :].mean(dim=1), p=2, dim=-1)

        need_transformer = (layer0_weight < 1.0 - 1e-6) or use_deepstack
        if need_transformer:
            pos_1d = (attn_mask.long().cumsum(-1) - 1).masked_fill(attn_mask == 0, 1)
            pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=pos_3d,
                output_hidden_states=True,
                return_dict=True,
            )
            if use_deepstack:
                ds_vecs = [
                    F.normalize(
                        outputs.hidden_states[l_idx + 1][:, start_idx:end_idx, :]
                        .to(torch.float32)
                        .mean(dim=1),
                        p=2,
                        dim=-1,
                    )
                    for l_idx in DEEPSTACK_LAYERS
                    if l_idx + 1 < len(outputs.hidden_states)
                ]
                e_ctx = F.normalize(torch.stack(ds_vecs, dim=0).mean(dim=0), p=2, dim=-1)
            else:
                layer1 = outputs.hidden_states[1].to(torch.float32)
                e_ctx = F.normalize(layer1[:, start_idx:end_idx, :].mean(dim=1), p=2, dim=-1)

            q_vec = F.normalize(layer0_weight * e0 + (1.0 - layer0_weight) * e_ctx, p=2, dim=-1)
        else:
            q_vec = e0

        scores = q_vec @ self.cal_embs_tensor.T
        return scores.squeeze(0).cpu().numpy()

    def _fast_personalized_pagerank(
        self,
        personalization_vec: np.ndarray,
        alpha: float = 0.85,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> dict[Any, float]:
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
        method: str = "1_layer0_boundary_pro",
        top_k_seeds: int = 1,
        k_neighbors: int = K_NEIGHBORS,
        smoothing_lambda: float = 0.35,
        verbose: bool = True,
    ) -> ClinicalTextRetrievalResult:
        t0 = time.perf_counter()

        if method not in self.methods_registry:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(self.methods_registry.keys())}"
            )

        raw_sims = self.methods_registry[method](text_query)

        smoothed_sims = np.copy(raw_sims)
        if self.G.number_of_edges() > 0 and smoothing_lambda > 0.0:
            nbr_means = self.smoothing_matrix @ raw_sims
            mask = self.has_neighbors_mask
            smoothed_sims[mask] = (
                (1.0 - smoothing_lambda) * raw_sims[mask]
                + (smoothing_lambda * nbr_means[mask])
            )

        sorted_indices = np.argsort(smoothed_sims)[::-1]
        seed_nodes = [self.node_list[i] for i in sorted_indices[:top_k_seeds]]

        sim_mean = float(smoothed_sims.mean())
        sim_std = max(float(smoothed_sims.std()), 1e-6)
        top_score = float(smoothed_sims[sorted_indices[0]])
        confidence_z = (top_score - sim_mean) / sim_std
        low_conf_warning = confidence_z < self.confidence_z_threshold

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
                max(
                    self.G[n][s].get("weight", 0.85)
                    for s in seed_nodes
                    if self.G.has_edge(n, s)
                )
                * ppr_scores.get(n, 0.0)
            ),
            reverse=True,
        )

        retrieved_subgraph = seed_nodes + ranked_neighbors[:k_neighbors]
        retrieved_systems = [self.get_node_info(n)[2] for n in retrieved_subgraph]
        consensus = dict(Counter(retrieved_systems))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if verbose:
            print("=" * 90)
            print(
                f"UN CLINICAL TEXT GRAPH RAG | METHOD: {method} | QUERY: '{text_query}' | "
                f"Latency: {latency_ms:.2f} ms"
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
            print("-" * 90)
            for node_id in retrieved_subgraph:
                node_idx = self.node_to_idx[node_id]
                role = "[SEED]" if node_id in seed_nodes else "[NBR] "
                img_name, clinical_snippet, body_sys = self.get_node_info(node_id)
                short_txt = clinical_snippet[:48] + "..." if len(clinical_snippet) > 48 else clinical_snippet
                print(
                    f"  {role} Node {str(node_id):<4} | System: {body_sys:<18} | "
                    f"Sim: {smoothed_sims[node_idx]:+.4f} | PPR: {ppr_scores[node_id]:.5f} | "
                    f"{img_name} | {short_txt}"
                )
            print("Subgraph Clinical Consensus:", consensus)

        return ClinicalTextRetrievalResult(
            query=text_query,
            method=method,
            seed_nodes=seed_nodes,
            retrieved_subgraph=retrieved_subgraph,
            smoothed_sims=smoothed_sims,
            ppr_scores=ppr_scores,
            confidence_z_score=confidence_z,
            is_low_confidence_warning=low_conf_warning,
            latency_ms=latency_ms,
            system_consensus=consensus,
        )


class TextGraphRAG:
    """
    High-level Clinical Text Graph RAG Service Wrapper.

    Mirrors `GraphRAG` from `vison_2.py` with safe lazy initialization and
    automatic cache invalidation when the graph instance changes.
    """

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(device) if device else get_default_device()
        self._cached_engine: UNClinicalTextGraphRAGEngine | None = None

    def query_text_graph_rag(
        self,
        text_query: str,
        G: nx.Graph,
        text_model: Qwen3VLTextModel,
        tokenizer: AutoTokenizer,
        text_embeddings: np.ndarray | None = None,
        method: str = "1_layer0_boundary_pro",
        top_k_seeds: int = 1,
        k_neighbors: int = K_NEIGHBORS,
        smoothing_lambda: float = 0.35,
        verbose: bool = True,
    ) -> tuple[list[Any], list[Any], np.ndarray]:
        """
        Drop-in entry point for Clinical Text Graph RAG retrieval.
        """
        global _CACHED_TEXT_ENGINE
        if self._cached_engine is None or self._cached_engine.G is not G:
            self._cached_engine = UNClinicalTextGraphRAGEngine(
                G=G,
                text_model=text_model,
                tokenizer=tokenizer,
                device=self.device,
                fallback_embs=text_embeddings,
            )
        _CACHED_TEXT_ENGINE = self._cached_engine

        res = self._cached_engine.query(
            text_query=text_query,
            method=method,
            top_k_seeds=top_k_seeds,
            k_neighbors=k_neighbors,
            smoothing_lambda=smoothing_lambda,
            verbose=verbose,
        )
        return res.seed_nodes, res.retrieved_subgraph, res.smoothed_sims