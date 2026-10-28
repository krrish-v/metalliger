"""
MetalLiger Utilities.

Provides vision and text multimodal processing and clinical Graph RAG modules.
"""

try:
    from .vison import (
        func,
        ClinicalRetrievalResult,
        UNClinicalGraphRAGEngine,
        GraphRAG,
    )
    from .text import (
        EmbedderConfig,
        TextEmbedder,
        LongContextEmbedder,
        PromptBuilder,
        PostProcessor,
        LexicalIndex,
    )
except ImportError:
    from metalliger.utils.vison import (
        func,
        ClinicalRetrievalResult,
        UNClinicalGraphRAGEngine,
        GraphRAG,
    )
    from metalliger.utils.text import (
        EmbedderConfig,
        TextEmbedder,
        LongContextEmbedder,
        PromptBuilder,
        PostProcessor,
        LexicalIndex,
    )

# Backward-compatibility aliases for legacy scripts
MedicalTextEmbedder = TextEmbedder
TextFunc = func
ClinicalTextRetrievalResult = ClinicalRetrievalResult
UNClinicalTextGraphRAGEngine = UNClinicalGraphRAGEngine
TextGraphRAG = GraphRAG

__all__ = [
    # Vision & Multimodal Graph RAG
    "func",
    "ClinicalRetrievalResult",
    "UNClinicalGraphRAGEngine",
    "GraphRAG",
    # Text Retrieval & Embedding
    "EmbedderConfig",
    "TextEmbedder",
    "MedicalTextEmbedder",
    "LongContextEmbedder",
    "PromptBuilder",
    "PostProcessor",
    "LexicalIndex",
    # Legacy aliases
    "TextFunc",
    "ClinicalTextRetrievalResult",
    "UNClinicalTextGraphRAGEngine",
    "TextGraphRAG",
]
