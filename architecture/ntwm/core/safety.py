"""
Safety mechanisms for the Neural Tropism World Model (NTWM).
Implements sequence sealing and export prevention.
"""

from typing import Dict, Any, Optional

class SafetyError(Exception):
    """Raised when safety constraints are violated."""
    pass

class SealedSequenceHandle:
    """Opaque handle for sealed sequences; raw sequence unavailable."""
    __slots__ = ("_meta",)
    
    def __init__(self, meta: Dict[str, Any]) -> None:
        self._meta = dict(meta)

    @property
    def meta(self) -> Dict[str, Any]:
        """Get metadata about the sealed sequence."""
        return dict(self._meta)

class SafetyGate:
    """Seals sequence payloads and prevents export of raw sequences."""
    
    def __init__(self, max_len: int = 1_000_000) -> None:
        self.max_len = max_len

    def seal(self, seq_bytes: Optional[bytes], serotype_class: Optional[str]) -> SealedSequenceHandle:
        """Seal a sequence payload into an opaque handle."""
        if seq_bytes is None and serotype_class is None:
            raise SafetyError("Empty sequence payload.")
        
        if seq_bytes is not None and len(seq_bytes) > self.max_len:
            raise SafetyError("Sequence too large.")
        
        meta = {
            "serotype_class": serotype_class,
            "has_bytes": seq_bytes is not None,
            "len": len(seq_bytes) if seq_bytes is not None else 0,
        }
        return SealedSequenceHandle(meta)

    def deny_export(self) -> None:
        """Prevent export of raw sequences."""
        raise SafetyError("Export of raw sequences is forbidden.")
