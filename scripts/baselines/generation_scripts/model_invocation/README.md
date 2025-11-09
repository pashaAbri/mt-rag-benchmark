# Model Invocation

This directory contains the shared LLM calling logic used by all generation scenarios (reference, reference_rag, full_rag).

## Structure

```
model_invocation/
├── llm_configs/           # Model configuration files
│   ├── llama_3.1_8b.yaml
│   └── README.md
├── llm_caller.py          # Main generation script
├── utils.py               # Helper functions (prompts, API calls)
└── README.md              # This file
```

## Usage

This module is called by scenario-specific scripts. It is not meant to be run directly.

### Example

```bash
python scripts/baselines/generation_scripts/model_invocation/llm_caller.py \
    --model_config scripts/baselines/generation_scripts/model_invocation/llm_configs/llama_3.1_8b.yaml \
    --input_file human/generation_tasks/reference.jsonl \
    --output_file scripts/baselines/generation_scripts/reference/results/llama_3.1_8b_reference.jsonl \
    --batch_size 10
```

## Configuration

Model configurations are stored in `llm_configs/`. See `llm_configs/README.md` for details.

## Environment Variables

- `TOGETHER_API_KEY`: API key for Together AI (required)

Set in your `.env` file in the project root.

## Components

### `llm_caller.py`

Main script that:
- Loads model configuration
- Processes tasks from input file
- Constructs prompts using paper format
- Calls LLM API
- Adds predictions to tasks
- Saves results with checkpointing

### `utils.py`

Helper functions:
- `construct_prompt()`: Build prompts from task data (paper Section D.2 format)
- `call_together_ai()`: Make API calls with retry logic
- `load_llm_config()`: Load YAML configurations
- `save_results_with_predictions()`: Save output files
- `load_existing_results()`: Support resume functionality

## Adding New Models

1. Create a new YAML config in `llm_configs/`
2. If using a different API provider, add provider-specific logic to `utils.py`
3. Test with a small sample before running on full dataset

