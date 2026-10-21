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
        TextFunc,
        ClinicalTextRetrievalResult,
        UNClinicalTextGraphRAGEngine,
        TextGraphRAG,
    )
except ImportError:
    from metalliger.utils.vison import (
        func,
        ClinicalRetrievalResult,
        UNClinicalGraphRAGEngine,
        GraphRAG,
    )
    from metalliger.utils.text import (
        TextFunc,
        ClinicalTextRetrievalResult,
        UNClinicalTextGraphRAGEngine,
        TextGraphRAG,
    )

__all__ = [
    "func",
    "ClinicalRetrievalResult",
    "UNClinicalGraphRAGEngine",
    "GraphRAG",
    "TextFunc",
    "ClinicalTextRetrievalResult",
    "UNClinicalTextGraphRAGEngine",
    "TextGraphRAG",
]
