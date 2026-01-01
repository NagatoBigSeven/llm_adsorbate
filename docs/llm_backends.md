# LLM Backend System

AdsKRK supports multiple LLM backends for the agentic workflow. Choose between cloud APIs (for best performance) or local models (for privacy and no API costs).

## Quick Start

The default backend is **Google AI (Gemini 2.5 Pro)**. Set your API key and run:

```bash
export GOOGLE_API_KEY="your-google-api-key"
streamlit run src/app/app.py
```

## Supported Backends

| Backend | Type | API Key Required | Best For |
|---------|------|------------------|----------|
| **Google AI** | Cloud | Yes (`GOOGLE_API_KEY`) | Production, low latency |
| **OpenRouter** | Cloud | Yes (`OPENROUTER_API_KEY`) | Access to multiple models |
| **Ollama** | Local | No | Privacy, offline use |
| **HuggingFace** | Local | No | Full customization |

## Backend Selection

### Via UI

1. Open the Streamlit app
2. In the sidebar, select your backend from the "🤖 LLM Backend" dropdown
3. Enter API key if using a cloud backend
4. Select your preferred model

### Via Environment Variable

```bash
# Use Google AI (default)
export ADSKRK_LLM_BACKEND=google
export GOOGLE_API_KEY="your-google-api-key"

# Use OpenRouter
export ADSKRK_LLM_BACKEND=openrouter
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Use Ollama (local)
export ADSKRK_LLM_BACKEND=ollama
# Make sure Ollama is running: ollama serve

# Use HuggingFace (local)
export ADSKRK_LLM_BACKEND=huggingface
export HF_QUANTIZE=4bit  # Optional: reduce memory usage
```

## Cloud Backends

### Google AI (Gemini)

Direct access to Google's Gemini models. Recommended for production use.

**Setup:**

1. Get API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set `GOOGLE_API_KEY` or enter in the app

**Available Models:**

- `gemini-2.5-pro` (default) - Best reasoning
- `gemini-2.5-flash` - Fast responses
- `gemini-2.5-flash-lite` - Fastest, lightweight

### OpenRouter

Access multiple AI providers through a unified API.

**Setup:**

1. Get API key from [OpenRouter](https://openrouter.ai)
2. Set `OPENROUTER_API_KEY` or enter in the app

**Available Models:**

- `google/gemini-3-pro-preview`
- `openai/gpt-5.2-pro`
- `anthropic/claude-opus-4.5`
- Any model from [OpenRouter's catalog](https://openrouter.ai/models)

## Local Backends

### Ollama

Run models locally using Ollama. Free and private.

**Setup:**

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3:8b

# Start the service
ollama serve
```

**Configuration:**

- `OLLAMA_HOST`: Server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Model Name (default: `qwen3:8b`)

### HuggingFace Transformers

Load models directly from HuggingFace Hub for full offline capability.

**Setup:**

```bash
# Install dependencies (included in project)
pip install transformers accelerate

# Optional: For quantization
pip install bitsandbytes
```

**Configuration:**

- `HF_MODEL`: Model ID (default: `Qwen/Qwen3-8B`)
- `HF_DEVICE`: Device (`auto`, `cuda`, `cpu`)
- `HF_QUANTIZE`: Quantization mode (`4bit`, `8bit`, `none`)

## Advanced Settings

The app provides advanced settings for fine-tuning LLM behavior:

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Temperature | 0.0 - 1.0 | 0.0 | Higher = more creative |
| Max Tokens | 256 - 16384 | 4096 | Maximum response length |

## Programmatic Usage

```python
from src.llms import get_llm_backend

# Get a backend
backend = get_llm_backend("google")

# Get default configuration
config = backend.get_default_config(api_key="your-api-key")

# Customize
config.model = "gemini-2.5-flash"
config.temperature = 0.3

# Get LangChain-compatible chat model
llm = backend.get_chat_model(config)
```

## Troubleshooting

### Ollama Connection Failed

```
❌ Ollama not running
```

**Solution:** Start Ollama with `ollama serve`

### HuggingFace Out of Memory

**Solution:** Enable quantization in the UI (select 4bit or 8bit in the Quantization dropdown) or via environment variable:

```bash
export HF_QUANTIZE=4bit
```

### API Key Not Found

**Solution:** Check environment variable or enter in the app sidebar.
