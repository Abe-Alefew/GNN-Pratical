# GNN + MLOps Assignments

This repository contains four progressively connected assignments: a GCN built from first principles, a Cora architecture comparison, leakage-safe link prediction, and a production-oriented MLOps serving pipeline. The project is code- and test-driven; notebook artifacts and notebook generators have been removed.

## Quick Start

```bash
# Create and activate an environment
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate              # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Make the repository importable
set PYTHONPATH=%cd%                     # Windows CMD
# export PYTHONPATH=$(pwd)               # Linux/macOS

# Run all assignment tests
pytest tests/ -v
```

The verified local baseline is **63 passed** tests. The suite currently emits three dependency deprecation warnings from FastAPI/Starlette/AnyIO; they do not fail the tests.

## Repository Structure

```text
configs/                 YAML configuration for A1-A4
src/common/              Seeds, metrics, config loading, data hashes
src/a1_gcn_scratch/      From-scratch dense and sparse GCN implementation
src/a2_cora/             MLP, GCN, GraphSAGE, GAT, tuning, experiments
src/a3_linkpred/         Edge splitting, negative sampling, leakage checks
src/a4_mlops/            Tracking, packaging, gate, drift, rollback, API
tests/                   Pytest suites for all four assignments
streamlit_app.py         Guided client for the Docker-hosted prediction API
Dockerfile               FastAPI inference image
REPORT.md                Detailed technical report and failure analysis
ISSUES.md                Root-cause and prevention log
```

## Assignment Commands

```bash
pytest tests/test_a1.py -v
pytest tests/test_a2.py -v
pytest tests/test_a3.py -v
pytest tests/test_a4.py -v
```

### Assignment CLI Commands

Run the implemented assignment entry points from the repository root:

```bash
# Assignment 1: train the from-scratch GCN and compare it with the MLP baseline
python -m src.a1_gcn_scratch.train

# Assignment 2: benchmark MLP, GCN, GraphSAGE, and GAT on Cora
python -m src.a2_cora.train

# Assignment 2: run depth and edge-removal experiments
python -m src.a2_cora.experiments

# Assignment 3: run leakage-safe link prediction experiments
python -m src.a3_linkpred.train --config configs/a3_config.yaml --seed 42 --verbose

# Assignment 3: override the configured training epoch count
python -m src.a3_linkpred.train --config configs/a3_config.yaml --epochs 50 --seed 42

# Assignment 4: train and track a model with MLflow
python -m src.a4_mlops.train_tracked

# Assignment 4: evaluate a candidate model for promotion
python -m src.a4_mlops.gate candidate_metrics.json incumbent_metrics.json

# Assignment 4: start the FastAPI server directly
uvicorn src.a4_mlops.api.app:app --host 0.0.0.0 --port 8000

# Assignment 4: run the end-to-end MLOps demonstration
python demo.py
```

The A1 and A2 training CLIs load their defaults from the implementation and Cora data layout. A3 supports `--config`, `--epochs`, `--seed`, and `--verbose`. A4 gate commands return exit code `0` when a candidate is promoted and `1` when promotion is blocked. Roll back the active champion with:

```bash
python -m src.a4_mlops.rollback "Regression detected in production"
```

Assignment 4 commands:

```bash
# Train with MLflow tracking
python -m src.a4_mlops.train_tracked

# Run the API locally
uvicorn src.a4_mlops.api.app:app --reload

# Build and run the API container
docker build -t gnn-mlops-api:latest .
docker run --rm -p 8000:8000 gnn-mlops-api:latest

# Run the guided Streamlit client in another terminal
streamlit run streamlit_app.py
```

Open `http://localhost:8501` for the Streamlit client. It reads `/model-info`, generates correctly sized graph requests, and provides controls for node count, feature strategy, seed, scale, topology, target nodes, API URL, and timeout. The advanced expander exposes the generated JSON for logging without requiring users to write matrices manually.

The API exposes:

- `GET /health` — service and model readiness
- `GET /model-info` — architecture, feature count, classes, and model metadata
- `POST /predict` — predictions for supplied nodes, edges, and target IDs

Each node must provide the model's full feature vector. The packaged Cora model expects **1,433 features per node**. The request must also include the target node's required graph neighborhood; partial graph context can change degree normalization and therefore predictions.

## Results and Interpretation

### Assignment 1: GCN from Scratch

The executed synthetic stochastic block model had 200 nodes, two communities, weak 3D features, 1,962 directed edges, and edge homophily 0.8889.

| Model or condition | Test accuracy | Test macro F1 |
|---|---:|---:|
| GCN, original graph | **96.0%** | **0.9600** |
| MLP, no edges | 61.0% | 0.6100 |

Interpretation: the observed GCN advantage was 35.0 percentage points. It comes from homophilous neighborhood information, not simply extra model capacity. The current A1 CLI does not print the full perturbation sweep. Dense and sparse implementations, manual normalization, and permutation equivariance are covered by tests.

### Assignment 2: Cora Architecture Comparison

Cora contains 2,708 papers, 1,433 binary features, seven classes, and strong citation homophily.

| Model | Test accuracy | Test macro F1 | Epoch time |
|---|---:|---:|---:|
| MLP | 56.1% | 0.5516 | 0.69 s |
| GCN | 80.2% | 0.7945 | 0.89 s |
| GraphSAGE | 78.7% | 0.7789 | 1.87 s |
| GAT | 80.9% | 0.7993 | 1.36 s |

Interpretation: graph structure added 24.1 points over the MLP in this single-seed run. GAT had the highest test accuracy at 80.9%, while GCN offered nearly the same accuracy with lower runtime. The depth CLI measured 73.4%, 80.2%, 81.3%, and 50.0% at depths 1, 2, 4, and 8. Edge removal reduced GCN from 80.2% to 56.5% at 100% removal; MLP stayed at 56.1%.

### Assignment 3: Link Prediction and Leakage Prevention

The implementation canonicalizes undirected pairs with `u < v`, splits those pairs, and only then creates the bidirectional training message-passing graph. Negative samples exclude every known positive edge.

| Method | Test ROC-AUC | Test AP |
|---|---:|---:|
| Common-neighbor baseline | 0.7180 | 0.7165 |
| Valid GCN plus dot-product decoder | 0.8818 | 0.8813 |
| Valid GCN plus symmetric MLP decoder | **0.9066** | **0.9057** |
| Deliberately leaky GCN | 0.9425 | 0.9378 |

Interpretation: the valid GNNs substantially improved over the heuristic baseline, with the symmetric MLP decoder strongest in this fixed-seed run. The leaky result remains invalid because held-out edges are available during message passing, turning evaluation into edge memorization. All seven automated leakage checks passed.

### Assignment 4: MLOps and Serving

The pipeline covers seeded training, MLflow metadata, model packages, validation gates, rollback, drift detection, and FastAPI serving. Model packages include configuration, weights, feature schema, labels, and metadata rather than a bare checkpoint.

The latest tracked run reported validation accuracy 0.7860, validation macro F1 0.7738, test accuracy 0.8100, test macro F1 0.8064, best epoch 11, and 1.58 seconds of training time. The end-to-end demo also verified exact reload invariance, gate promotion/blocking, rollback, HTTP 200 serving, HTTP 422 validation errors, unseen-node inference, and drift alerting after a perturbed graph snapshot.

Validated behaviors include:

- Exact model reload prediction equivalence
- First-model promotion and regression blocking
- Threshold and tie handling
- Missing and NaN metric fail-closed behavior
- Health and model metadata endpoints
- Successful prediction requests
- Wrong feature dimensions, invalid edges, unknown targets, duplicate targets, and isolated nodes
- Champion registration, promotion, and rollback
- Graph and embedding drift checks

Interpretation: packaging and gates make model promotion auditable and reversible. The API is intentionally graph-context aware: a target prediction depends on the supplied neighborhood, and subgraph degree normalization can differ from full-graph offline inference. The Streamlit client makes this contract usable without hand-authoring large feature matrices.

## CI

GitHub Actions is defined in `.github/workflows/ci.yml` for Python 3.11. It caches pip dependencies, installs `requirements.txt`, sets `PYTHONPATH`, runs all tests with short tracebacks, and executes the first-model validation gate. The CI workflow is notebook-free and should also compile the application entry points:

```bash
python -m py_compile demo.py streamlit_app.py
```

The Docker image is built locally or in a release job after model artifacts are supplied. `model_packages/` is intentionally ignored because weights are generated artifacts and are not committed to source control; a CI Docker build therefore requires an artifact download or package-generation step first.

## Operational Notes

- Run the API before opening the Streamlit prediction console.
- Use `/model-info` to confirm the required feature dimension before sending a request.
- Generated zero, one-hot, and random feature vectors are shape-valid smoke-test inputs, not replacements for real Cora features.
- An unobserved link prediction is a ranked candidate, not proof that a citation exists.
- See [REPORT.md](REPORT.md) for the full derivations, empirical tables, and debugging analysis.