# NTWM Documentation

This directory contains comprehensive documentation for the Neural Tropism World Model (NTWM).

## **Reading Order to Learn the Codebase**

1. **Start Here** → `ntwm/core/types.py` (understand data structures)
2. **Safety** → `ntwm/core/safety.py` (security mechanisms)
3. **Models** → `ntwm/models/` (neural network components)
4. **Examples** → `ntwm/examples/demo.py` (usage patterns)
5. **Tests** → `tests/` (verify understanding)


## **Design Principles**

### **Safety-First Design**
- Sequences are sealed and never exposed
- Export mechanisms are blocked by design
- All interfaces are export-safe

### **Modular Architecture**
- Clear separation of concerns
- Independent component testing
- Easy to extend and modify

### **Research Focus**
- Designed for experimentation
- Clean interfaces for research
- Comprehensive type safety


