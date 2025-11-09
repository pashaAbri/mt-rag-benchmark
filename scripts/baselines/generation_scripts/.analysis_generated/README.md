# MT-RAG Analysis Scripts

## Quick Start

```bash
# Step 1: Calculate all metrics from your experimental results
python3 calculate_results.py

# Step 2: Generate tables (your results only)
python3 generate_tables.py

# Step 3: Generate combined tables (paper + your results)
python3 generate_combined_tables.py
```

## Output Files

After running all scripts:

### Your Results Only
- `aggregated_results.json` - All calculated metrics (raw data)
- `table_5_generated.md` - Your results by retrieval setting
- `table_16a_answerability_generated.md` - Your results by answerability
- `table_16b_turns_generated.md` - Your results by turn position
- `table_16c_domain_generated.md` - Your results by domain

### Combined (Paper + Experimental)
- `table_5_combined.md` - Paper vs. Experimental comparison
- `table_16a_combined.md` - Paper vs. Experimental (answerability)
- `table_16b_combined.md` - Paper vs. Experimental (turns)
- `table_16c_combined.md` - Paper vs. Experimental (domains)

## Scripts

1. **`calculate_results.py`**: Loads JSONL files, calculates averages, saves to JSON
2. **`generate_tables.py`**: Generates tables from your results only
3. **`generate_combined_tables.py`**: Generates side-by-side comparison tables

## Data Sources

- **Paper results**: `../.analysis_from_paper/paper_results.json` (n=426 tasks)
- **Your results**: `aggregated_results.json` (n=842 tasks)

## Color Coding

- **Paper tables**: Blue (#0066CC)
- **Your tables**: Green bold (#009900)
- **Combined tables**: Blue (paper) / Green bold (yours)

## Input Data

Scripts read from:
- `../reference/results/*_evaluated.jsonl` (842 tasks)
- `../reference_rag/results/*_evaluated.jsonl` (436 tasks)
- `../full_rag/results/*_evaluated.jsonl` (842 tasks)

## Metrics

- **Ans. Acc.**: Answerability accuracy
- **RLF**: Reference-Less Faithfulness
- **RBllm**: Reference-Based LLM judge
- **RBalg**: Reference-Based Algorithmic

