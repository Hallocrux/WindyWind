from __future__ import annotations

try:
    from annotate_ui import AnnotationSession, AnnotationState, AnnotationStore
except ModuleNotFoundError:
    from src.windNotFound.annotate_ui import AnnotationSession, AnnotationState, AnnotationStore

__all__ = ["AnnotationSession", "AnnotationState", "AnnotationStore"]
