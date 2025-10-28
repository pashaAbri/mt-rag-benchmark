# LLM Configuration Files

This directory contains YAML configuration files for different LLM models used in RAG generation experiments.

## Configuration Format

Each YAML file should contain:

```yaml
model_name: "descriptive_model_name"
provider: "api_provider_name"
api_model_id: "provider/model-id"
api_endpoint: "https://api.example.com/endpoint"
max_tokens: 200
temperature: 0.0
top_p: 1.0
do_sample: false
```

## Available Models

All models below are configured to match the MTRAG paper baseline experiments (Section 6.2).

### Llama 3.1 Family (Meta) - Together AI

**Llama 3.1 8B** - `llama_3.1_8b.yaml` ✓ Tested
- Parameters: 8B
- Context: 128K tokens
- Expected RBalg (Reference): 0.36

**Llama 3.1 70B** - `llama_3.1_70b.yaml`
- Parameters: 70B
- Context: 128K tokens
- Expected RBalg (Reference): 0.44

**Llama 3.1 405B** - `llama_3.1_405b.yaml`
- Parameters: 405B (Best open-source)
- Context: 128K tokens
- Expected RBalg (Reference): 0.48

### Qwen 2.5 Family (Alibaba) - Together AI

**Qwen 2.5 7B** - `qwen_2.5_7b.yaml`
- Parameters: 7B
- Context: 128K tokens
- Expected RBalg (Reference): 0.43

**Qwen 2.5 72B** - `qwen_2.5_72b.yaml`
- Parameters: 72B
- Context: 128K tokens
- Expected RBalg (Reference): 0.44
- Note: Competitive in noisy RAG settings

### Mixtral (Mistral AI) - Together AI

**Mixtral 8x22B** - `mixtral_8x22b.yaml`
- Parameters: 8x22B (Mixture of Experts)
- Context: 32K tokens
- Expected RBalg (Reference): 0.41

### Command R+ (Cohere) - Together AI

**Command R+ 104B** - `command_r_plus.yaml`
- Parameters: 104B
- Context: Not specified
- Expected RBalg (Reference): 0.44
- Note: Optimized for RAG and tool use

### GPT-4o Family (OpenAI) - OpenAI API

**GPT-4o** - `gpt4o.yaml`
- Best performing model (tied with Llama 405B)
- Context: 128K tokens
- Expected RBalg (Reference): 0.45
- **Requires:** OpenAI API key (OPENAI_API_KEY)
- **Requires:** Adding `call_openai()` function to `utils.py`

**GPT-4o-mini** - `gpt4o_mini.yaml`
- Cheaper alternative with good performance
- Context: 128K tokens
- Expected RBalg (Reference): 0.43
- **Requires:** OpenAI API key (OPENAI_API_KEY)
- **Requires:** Adding `call_openai()` function to `utils.py`

---

## API Keys

**Together AI Models:**
- Environment variable: `TOGETHER_API_KEY`
- Add to `.env` file in project root

**OpenAI Models:**
- Environment variable: `OPENAI_API_KEY`
- Add to `.env` file in project root
- Requires code modification to support OpenAI API

## Adding New Models

To add a new model:

1. Create a new YAML file (e.g., `gpt4o_mini.yaml`)
2. Fill in the configuration parameters
3. Update the provider-specific API call logic in `utils.py` if needed
4. Test with a small subset of tasks

## Environment Variables

- `TOGETHER_API_KEY`: API key for Together AI
- `OPENAI_API_KEY`: API key for OpenAI models (if added)
- `ANTHROPIC_API_KEY`: API key for Anthropic models (if added)

