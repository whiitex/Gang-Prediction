# Gang Prediction - Copilot Instructions

## Project Overview
Research codebase implementing **coarsening-aware GNN training** on graph data (Cora dataset). The core innovation: progressively coarsen the graph across training levels while maintaining prediction accuracy through custom loss functions that preserve spectral properties and node embeddings.

## Architecture & Data Flow

### Three-Layer System
1. **Graph Coarsening Layer** ([coarsening_utils.py](../src/coarsening_utils.py))
   - Implements variation-based graph coarsening (Loukas 2020 method)
   - Key methods: `variation_embedding`, `variation_edges`, `variation_neighborhoods`
   - Produces coarsening matrices `C` (coarse-to-fine) and `C_plus` (fine-to-coarse projection)
   - Uses spectral basis `B` computed via Laplacian eigenvectors

2. **GNN Training Layer** ([train_GNN_coarsening.py](../src/train_GNN_coarsening.py), [GNN_model.py](../src/GNN_model.py))
   - 2-layer SAGEConv GCN model (not standard GCNConv)
   - Trains iteratively across coarsening levels (multi-level training)
   - Projects predictions back to original graph: `pred_fine = C_plus @ logits`

3. **Loss Function** ([coarsening_aware_loss.py](../src/coarsening_aware_loss.py))
   - Combines classification loss with embedding regularization
   - At coarse levels: adds negative L2 norm penalty + negative sampling contrastive term
   - Prevents embeddings from collapsing during aggressive coarsening

### Critical Data Structures
- **PyG Data object** extended with custom attributes:
  - `data.W`: sparse adjacency matrix (from `graph_params`)
  - `data.L`: graph Laplacian
  - `data.S_mp`: message-passing edge structure (used in forward pass)
  - `data.C`, `data.C_plus`: coarsening projection matrices
  - `data.embeddings`: L2-normalized node embeddings from GNN
  - `data.soft_y`: one-hot encoded labels for soft aggregation

## Key Workflows

### Running Experiments
```bash
conda activate FedStruct  # Required environment
cd "Gang-Prediction"
python src/test.py        # Main experiment driver
```

**Entry point**: [src/test.py](../src/test.py) loads Cora dataset, runs vanilla GNN baseline, then iterates over coarsening thresholds and epochs-per-level configurations.

**Results structure**: Timestamped folders in `results/YYYYMMDD_HHMMSS/` contain:
- `.npy` files with training metrics (accuracy, loss, node counts per level)
- Naming: `data_gnn_CoarseningAwareLoss_V2_th_{threshold}_epochs_{epochs}.npy`

### Adding New Coarsening Methods
1. Implement coarsening logic in `coarse_one_level()` ([coarsening_utils.py](../src/coarsening_utils.py#L95))
2. Return coarsening list (list of node pairs to merge)
3. Update `method` parameter in [train_GNN_coarsening_aware_loss()](../src/train_GNN_coarsening.py#L67)

## Project-Specific Conventions

### Sparse Matrix Handling
- Always use **torch sparse COO tensors**, never scipy sparse
- Helper: `sparse_eye(size)` creates identity matrix ([utils/utils.py](../src/utils/utils.py#L96))
- Graph parameters computed via `graph_params(data)` which returns `(W, L, dw)` tuple
- Multiply sparse matrices with: `torch.sparse.mm(A, B)`

### Model Training Pattern
```python
# Standard training loop structure:
model.train()
logits = model(data.x, S_mp)  # Note: uses S_mp not edge_index!
if C_plus is not None:
    pred_fine = F.softmax(C_plus @ logits, dim=1)  # Project to fine level
loss = criterion(pred_fine, labels, train_idx, embeddings, coarse_loss=True)
```

### Device Management
- Global device set in [utils/utils.py](../src/utils/utils.py#L29): `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- All tensors created with `.to(device)` explicitly
- Model moved to device in training functions

### Path Conventions
- All imports use: `sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))`
- Run scripts from **project root**, not `src/` directory
- Results auto-saved with timestamp: `results/{datetime.now()}/`

## Common Gotchas

1. **S_mp vs edge_index**: Model's forward pass uses `data.S_mp` (weighted message-passing structure), not raw `edge_index`. Check for `hasattr(data, 'S_mp')` before falling back to `data.W`.

2. **Coarsening ratio calculation**: The `r_cur` parameter in coarsening is dynamically computed per level:
   ```python
   ratio = np.log(level ** (4/3)) / 100 + 0.01  # Grows logarithmically
   ```

3. **Label aggregation**: When coarsening, use `data.soft_y` (one-hot) not `data.y` (class indices) for proper soft label assignment to supernodes.

4. **Embedding normalization**: Always L2-normalize embeddings before coarsening:
   ```python
   Gc.embeddings = F.normalize(embeddings, p=2, dim=1)
   ```

5. **Evaluation on original graph**: After training on coarse graph, always project logits back:
   - Coarse accuracy: evaluate on `Gc` directly
   - Fine accuracy: project via `C_plus @ logits` then evaluate on original `data`

## Dependencies & Environment
- PyTorch Geometric (PyG) for graph neural networks
- pygsp for graph signal processing utilities
- sortedcontainers for efficient candidate management in coarsening
- Conda environment: `FedStruct` (pre-configured, no requirements.txt provided)

## File Organization
- **Core logic**: `src/train_GNN_coarsening.py`, `src/coarsening_utils.py`
- **Model definitions**: `src/GNN_model.py`, `src/coarsening_aware_loss.py`
- **Utilities**: `src/utils/utils.py` (data prep, sparse ops), `src/utils/visualization.py` (plotting)
- **Experiments**: `src/test.py` (main driver)
- **Scratch files**: `src/temp.py`, `src/temp3.py` (experimental sparse algebra helpers)
- **Data**: Cora dataset cached in `data/Planetoid/Cora/`
