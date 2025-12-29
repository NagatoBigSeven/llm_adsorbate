<p style="text-align: center;">
  <img src="./assets/adskrk_concept.png" alt="Project logo" width="80%"/>
</p>

<br>

[![Cookiecutter template from @SchwallerGroup](https://img.shields.io/badge/Cookiecutter-schwallergroup-blue)](https://github.com/schwallergroup/liac-repo)
[![Learn more @SchwallerGroup](https://img.shields.io/badge/Learn%20%0Amore-schwallergroup-blue)](https://schwallergroup.github.io)

<h1 align="left">
  AdsKRK: An agentic atomistic simulation framework for surface science.
</h1>

<br>

Welcome to the AdsKRK repository! This project is a prototype developed during the 2-day [LLM Hackathon for Application in Chemistry and Materials Science](https://llmhackathon.github.io/).

The goal of AdsKRK is to showcase how Large Language Models (LLMs) can autonomously explore the binding configurations of adsorbates on hetero-catalytic surfaces. Starting from only a SMILES string and a surface structure, the agent can:

* generate binding configurations,
* run structure relaxations,
* analyze the results, and
* iterate until a stable configuration is found.

Users can also interact with the agent - asking questions about the system or guiding the search process through prompts.

At the core of AdsKRK is [AutoAdsorbate](https://github.com/basf/autoadsorbate) - a powerful tool for generating chemically meaningful molecular and fragment configurations on surfaces, providing a search space for the agent.

## ✨ Features

* **Multi-Backend LLM Support**: Google AI (Gemini), OpenRouter, Ollama, HuggingFace
* **Multiple Structure Formats**: XYZ, CIF, PDB, SDF, MOL, POSCAR/VASP
* **Interactive UI**: Streamlit-based interface with real-time agent feedback
* **Local & Cloud Options**: Use cloud APIs or run completely offline with Ollama/HuggingFace

## 🚀 Quickstart

```bash
# Clone and install
git clone https://github.com/schwallergroup/llm_adsorbate.git
cd llm_adsorbate
uv pip install -e .

# Set API key (Google AI is default)
export GOOGLE_API_KEY="your-api-key"

# Run the app
streamlit run src/app/app.py
```

Then provide your inputs in the sidebar:

1. **SMILES**: Molecule structure (e.g., `CO2`, `H2O`)
2. **Slab File**: Upload your surface structure
3. **Query**: What you want to find
4. Click **▶️ Run**

📖 See [Quickstart Guide](docs/quickstart.md) for detailed instructions.

## 🤖 LLM Backend Options

| Backend | Type | API Key | Best For |
|---------|------|---------|----------|
| **Google AI** | Cloud | `GOOGLE_API_KEY` | Production (default) |
| **OpenRouter** | Cloud | `OPENROUTER_API_KEY` | Multi-model access |
| **Ollama** | Local | Not needed | Privacy, offline |
| **HuggingFace** | Local | Not needed | Customization |

Select your backend in the app sidebar or via environment variable:

```bash
export ADSKRK_LLM_BACKEND=google    # or openrouter, ollama, huggingface
```

📖 See [LLM Backends](docs/llm_backends.md) for configuration details.

## 📁 Supported File Formats

| Format | Extensions |
|--------|------------|
| XYZ | `.xyz` |
| CIF | `.cif` |
| PDB | `.pdb` |
| SDF/MOL | `.sdf`, `.mol` |
| POSCAR | `.poscar`, `.vasp` |

## ⚙️ Configuration

### API Keys

Multiple ways to provide your API key (in priority order):

1. **Environment variable**:

   ```bash
   export GOOGLE_API_KEY="your-key"
   # or OPENROUTER_API_KEY for OpenRouter
   ```

2. **Config file**: Use the app's "Save for future sessions" checkbox
   * Stored at: `~/.adskrk/config.json`

3. **Manual input**: Enter in the sidebar each session

### Advanced Settings

The app provides advanced settings (click ⚙️ Advanced Settings):
* **Temperature**: 0.0 (deterministic) to 1.0 (creative)
* **Max Tokens**: 256 to 16384

## 🔬 Example: CO₂ on Copper

One particularly interesting finding was the agent's ability to reason about relaxation trajectories. For CO₂ on a copper surface, Gemini 2.5 Pro can analyze:

```
The stability of the initial adsorption configuration was assessed by 
performing a structural relaxation. Based on the output from the simulation, 
the fragment did not remain bound to the surface.
...
Therefore, to answer the user's question: no, the fragment does not stay 
covalently bound. The initial configuration, with the carbon atom placed 
on a top site of the Cu(211) surface, is unstable and leads to desorption.
```

## 📚 Documentation

* [Quickstart Guide](docs/quickstart.md) - Get started in 5 minutes
* [LLM Backends](docs/llm_backends.md) - Configure LLM providers
* [Calculator Backends](docs/calculator_backends.md) - MACE and other calculators

## 👩‍💻 Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

* [AutoAdsorbate](https://github.com/basf/autoadsorbate) - Surface configuration generation
* [MACE](https://github.com/ACEsuit/mace) - Machine learning interatomic potentials
* [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
