# Polaris
A Neural Tropism World Model (NTWM) for predicting AAV capsid delivery in specified cells.

A clean, modular architecture for RNA/DNA delivery modeling with safety-focused design.

## Architecture Overview

The NTWM is organized into clear, focused modules that handle specific concerns:

### Core Components

- `types.py`: all dataclasses and type definitions
- `safety.py`: safety gate and sequence handling mechanisms
- `encoders.py`: frozen sequence encoders (nucleotide/protein)
- `priors.py`: priors gateway and utility functions
- `tokenizer.py`: spatiotemporal tokenizer with VQ-VAE
- `dynamics.py`: inverse dynamics and ST dynamics models
- `decoder.py`: decoder and explainer for overlays
- `utils.py`: utility functions and demo helpers

### Key Design Principles (for Cursor)

1. Separation of Concerns: each module has a single, clear responsibility
2. Safety First: sequence data is sealed and export is prevented
3. Modularity: components can be used independently or together
4. Type Safety: comprehensive type hints and dataclasses
5. Clean Interfaces: simple, well-documented public APIs

## Usage

```python
from ntwm import (
    SpatiotemporalTokenizer,
    STDynamics,
    DecoderExplainer,
    dummy_contexts
)

# Initialize models
tok = SpatiotemporalTokenizer(channels=8, H=32, W=32, K=1024)
dyn = STDynamics(K=1024, H=32, W=32)
dec = DecoderExplainer(tok.codebook, H=32, W=32)

# Use components...
```

## Safety

- Sequence Sealing: raw sequences are never exposed
- Export Prevention: safety gates block unauthorized access
- Non-diff Priors: coarse, discrete properties only
- Opaque Handles: metadata-only access to sequences
