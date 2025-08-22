"""
Neural Tropism World Model (NTWM) - Modular architecture.

This package provides a modular architecture for RNA/DNA delivery modeling
with clear separation of concerns and interfaces.
"""

# Core components
from .core import (
    # Types
    CellContext,
    TissueContext, 
    OmicsContext,
    StructuralPrior,
    RegulatoryPrior,
    MiscPrior,
    SequencePrior,
    Overlay,
    Tensor,
    
    # Utils
    dummy_contexts
)

# Neural network models
from .models import (
    # Encoders
    FrozenNucEncoder,
    FrozenProtEncoder,
    
    # Tokenizer
    SpatiotemporalTokenizer,
    VQEncoder,
    VQCodebook, 
    VQDecoder,
    
    # Dynamics
    STDynamics,
    InverseDynamics,
    ActionVQ,
    
    # Decoder
    DecoderExplainer
)

# Priors and context handling
from .priors import (
    PriorsGateway,
    pack_priors
)

# Examples
from .examples import main as run_demo

__version__ = "0.1.0"
__all__ = [
    # Core
    "CellContext",
    "TissueContext", 
    "OmicsContext",
    "StructuralPrior",
    "RegulatoryPrior",
    "MiscPrior",
    "SequencePrior",
    "Overlay",
    "Tensor",
    "dummy_contexts",
    
    # Models
    "FrozenNucEncoder",
    "FrozenProtEncoder",
    "SpatiotemporalTokenizer",
    "VQEncoder",
    "VQCodebook", 
    "VQDecoder",
    "STDynamics",
    "InverseDynamics",
    "ActionVQ",
    "DecoderExplainer",
    
    # Priors
    "PriorsGateway",
    "pack_priors",
    
    # Examples
    "run_demo"
]
