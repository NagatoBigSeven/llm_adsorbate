# Quickstart Guide

Get AdsKRK running in 5 minutes.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- API key from Google AI or OpenRouter (for cloud backends)

## Installation

```bash
# Clone the repository
git clone https://github.com/schwallergroup/llm_adsorbate.git
cd llm_adsorbate

# Install dependencies
uv pip install -e .
```

## Step 1: Set Up LLM Backend

### Option A: Google AI (Recommended)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set the environment variable:

   ```bash
   export GOOGLE_API_KEY="your-google-api-key"
   ```

### Option B: OpenRouter

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Set the environment variable:

   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   ```

### Option C: Ollama (Local, No API Key)

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull a model: `ollama pull qwen3:8b`
3. Start the service: `ollama serve`

## Step 2: Launch the App

```bash
streamlit run src/app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Step 3: Run Your First Simulation

### 1. Select LLM Backend (Sidebar)

- Choose your backend (Google AI, OpenRouter, Ollama, or HuggingFace)
- Enter API key if using a cloud backend
- Select a model

### 2. Enter Inputs

| Input | Description | Example |
|-------|-------------|---------|
| **SMILES** | Molecule/ReactiveSpecies | `O=C=O`, `O`, `CO` |
| **Slab File** | Surface structure file | Upload XYZ, CIF, or POSCAR |
| **Query** | What you want to find | "Find the most stable adsorption site" |

### 3. Click "▶️ Run"

The agent will:

1. Parse your adsorbate and surface structure file
2. Propose an adsorption configuration based on chemical reasoning
3. Populate the adsorbate onto the surface using AutoAdsorbate
4. Run relaxation simulation with MACE
5. Analyze results and iterate if needed
6. Report the best configuration found

## Example: CO₂ on Copper

**Inputs:**

- SMILES: `O=C=O`
- Slab: `notebooks/cu_slab_211.xyz` (included in repo)
- Query: "Find the most stable binding configuration for CO2"

**What happens:**

1. Agent analyzes surface structure and identifies binding site types
2. Proposes a binding configuration based on chemical reasoning
3. Runs MACE relaxation to test stability
4. Analyzes results — if unstable, proposes a different configuration
5. Iterates until finding a stable configuration
6. Reports findings with structural analysis

## Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| XYZ | `.xyz` | Standard XYZ coordinates |
| CIF | `.cif` | Crystallographic Information File |
| PDB | `.pdb` | Protein Data Bank format |
| SDF/MOL | `.sdf`, `.mol` | MDL Molfile format |
| POSCAR | `.poscar`, `.vasp` | VASP structure format |

## Tips

- **Start simple**: Use small molecules like CO2, H2O, or CH4
- **Be specific**: Clear queries get better results
- **Check status**: The agent shows its reasoning in real-time
- **Use Clear button**: Reset conversation between experiments

## Next Steps

- [LLM Backend Configuration](llm_backends.md) - Advanced LLM settings
- [Calculator Backends](calculator_backends.md) - MACE and other calculators
