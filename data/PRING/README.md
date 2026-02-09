# PRING Dataset

Processed protein-protein interaction (PPI) dataset from [PRING](https://huggingface.co/datasets/piaolaidangqu/PRING).

**Species:** Human, Yeast, Arabidopsis thaliana (arath), E. coli

## Data Structure Overview

```
PRING/
├── human/                          # Human (H. sapiens) - Primary training species
│   ├── BFS/                        # Breadth-First Search split strategy
│   ├── DFS/                        # Depth-First Search split strategy
│   ├── RANDOM_WALK/                # Random Walk split strategy
│   ├── human.fasta                 # Full FASTA with metadata
│   ├── human_simple.fasta          # Simple FASTA (UniProt IDs only)
│   ├── human_ppi.txt               # All PPI pairs (129,861 pairs)
│   ├── human_protein_id.csv        # Protein metadata table
│   └── human_graph.pkl             # Full PPI network graph
├── yeast/                          # Yeast (S. cerevisiae) - Cross-species eval
├── arath/                          # Arabidopsis thaliana - Cross-species eval
├── ecoli/                          # E. coli - Cross-species eval
└── species_processed_data/         # Alternative processed versions
    ├── human/
    ├── yeast/
    ├── arath/
    └── ecoli/
```

## Dataset Statistics

### Human (Primary Training Data)
- **Proteins:** 10,289 unique proteins
- **Total PPIs:** 129,861 pairs
- **Splits (BFS example):**
  - Train: 85,824 pairs (66.1%)
  - Validation: 21,456 pairs (16.5%)
  - Test: 64,038 pairs (49.3% - overlapping evaluation)
  - All-test ranking: 2,036,358 pairs (all-vs-all for test proteins)

### Cross-Species (Zero-shot Evaluation)
| Species | Proteins | PPIs | Test PPIs | All-test Pairs |
|---------|----------|------|-----------|----------------|
| **Yeast** | 3,274 | 15,921 | 15,176 | 500,500 |
| **Arath** | ~5,600 | ~24,000 | ~14,000 | 500,500 |
| **E. coli** | ~2,500 | ~17,000 | ~13,000 | 500,500 |

## File Format Specifications

### 1. PPI Files (`*_ppi.txt`)
**Format:** Tab-separated values (TSV)
```
P05120  P00749  1
P51398  Q9BQC3  0
```
- **Column 1:** UniProt ID of Protein A
- **Column 2:** UniProt ID of Protein B
- **Column 3:** Label (1 = positive interaction, 0 = negative/non-interaction)

### 2. FASTA Files (`*.fasta`, `*_simple.fasta`)
**Format:** Standard FASTA
```
>P25786
MFRNQYDNDVTVWSPQGRIHQIEYAMEAVKQGSAT...
```
- **Header:** UniProt ID (simple) or full metadata (regular)
- **Sequence:** Amino acid sequence (single-letter code)

### 3. Protein Metadata (`*_protein_id.csv`)
**Format:** CSV with header
```csv
uniprot_id,organism_id,sequence,sequence_length
P25786,9606,MFRNQYDND...,263
```
- **uniprot_id:** UniProt accession
- **organism_id:** NCBI Taxonomy ID (9606=Human, 559292=Yeast, etc.)
- **sequence:** Full amino acid sequence
- **sequence_length:** Number of residues

### 4. Graph Files (`*_graph.pkl`, `*_test_graph.pkl`)
**Format:** Pickled NetworkX Graph
- **Nodes:** UniProt IDs
- **Edges:** Protein-protein interactions
- **Usage:** Network topology analysis, GNN training

### 5. Sampled Nodes (`*_sampled_nodes.pkl`)
**Format:** Pickled list/dict of node subgraphs
- **Purpose:** Pre-computed subgraph samples for graph-level testing
- **Variants:** BFS/DFS/RANDOM_WALK-specific sampling

### 6. Split Metadata (`human_*_split.pkl`)
**Format:** Pickled dictionary
- **Contains:** Train/val/test indices and metadata for reproducibility

## Split Strategies (Human Only)

### BFS (Breadth-First Search)
- Nodes selected via BFS traversal from random seeds
- **Goal:** Preserve local network structure in splits
- **Use case:** Testing generalization to nearby network regions

### DFS (Depth-First Search)
- Nodes selected via DFS traversal
- **Goal:** Create elongated, path-like test sets
- **Use case:** Evaluating long-range dependencies

### RANDOM_WALK
- Nodes selected via random walk sampling
- **Goal:** Stochastic exploration of network structure
- **Use case:** Balanced structural diversity

### Split Files per Strategy

Each split directory (`BFS/`, `DFS/`, `RANDOM_WALK/`) contains:
- `human_train_ppi.txt` - Training pairs
- `human_val_ppi.txt` - Validation pairs
- `human_test_ppi.txt` - Test pairs (binary classification)
- `all_test_ppi.txt` - All-vs-all test protein combinations (ranking task)
- `human_train_graph.pkl` - Training subgraph
- `human_test_graph.pkl` - Test subgraph
- `test_sampled_nodes.pkl` - Precomputed test subgraphs
- `human_*_split.pkl` - Split metadata

## Cross-Species Evaluation (Yeast, Arath, E. coli)

These species are used for **zero-shot cross-species transfer** evaluation:
- No train/val splits provided
- `*_test_ppi.txt` - Positive-labeled test pairs
- `*_all_test_ppi.txt` - All-vs-all pairs for ranking (500,500 pairs = 1000×1000 proteins)
- `*_{BFS,DFS,RANDOM_WALK}_sampled_nodes.pkl` - Graph-level test samples aligned with human split strategies

## Preprocessing Pipeline

### Sequence Filtering
1. **Length:** 50–1,000 amino acids
2. **Sequence similarity:** <40% identity (MMseqs2 clustering)
3. **Functional redundancy:** Removed across species to prevent data leakage

### Negative Sampling
- Non-interacting pairs sampled from PPI network
- Balanced positive:negative ratio (~1:1)

## Usage Notes

### For PPI Prediction (Binary Classification)
- Use `*_train_ppi.txt`, `*_val_ppi.txt`, `*_test_ppi.txt`
- Labels: 1=interaction, 0=non-interaction

### For Ranking Tasks
- Use `all_test_ppi.txt` (all-vs-all combinations)
- Evaluate retrieval metrics (e.g., precision@k, MRR)

### For Cross-Species Transfer
- Train on human splits
- Evaluate on yeast/arath/ecoli test sets
- Tests zero-shot generalization across organisms

## Data Provenance
- **Source:** [PRING HuggingFace Dataset](https://huggingface.co/datasets/piaolaidangqu/PRING)
- **Original PPI databases:** STRING, BioGRID, IntAct
- **Sequence source:** UniProt
- **Reference:** See PRING publication for detailed methodology
