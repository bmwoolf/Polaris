# Polaris
A Neural Tropism World Model (NTWM) for predicting AAV capsid delivery in specified cells.

A clean, modular architecture for RNA/DNA delivery modeling with clear separation of concerns.

## Architecture Overview

The NTWM is organized into clear, focused modules that handle specific concerns:

### Core Components

- `types.py`: all dataclasses and type definitions
- `encoders.py`: frozen sequence encoders (nucleotide/protein)
- `priors.py`: priors gateway and utility functions
- `tokenizer.py`: spatiotemporal tokenizer with VQ-VAE
- `dynamics.py`: inverse dynamics and ST dynamics models
- `decoder.py`: decoder and explainer for overlays
- `utils.py`: utility functions and demo helpers

### Key Design Principles (for Cursor)

1. Separation of Concerns: each module has a single, clear responsibility
2. Modularity: components can be used independently or together
3. Type Safety: comprehensive type hints and dataclasses
4. Clean Interfaces: simple, well-documented public APIs

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

## Features

- Sequence processing without safety constraints
- Frozen encoders (no parameter updates)
- Non-differentiable priors
- Export-safe outputs only
- No raw sequence access anywhere

## Limitations

- No sequence generation - can't create new DNA/RNA sequences
- No optimization - can't say "make me a better capsid sequence"
- No modification - can't edit or change existing sequences
- Research only - system only analyzes and predicts, doesn't create
