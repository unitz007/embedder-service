# How to Analyze Any Repository with Full Context LLM

## Overview
The `analyze_any_repo.py` script allows you to analyze any local repository using the full context LLM approach. It gives the LLM complete access to the repository structure, configuration files, and relevant code sections while using embeddings for efficient semantic search.

## Prerequisites
1. Ollama running: `ollama serve` (in background)
2. At least one model pulled: `ollama pull gemma3:1b` (or your preferred model)
3. The repository you want to analyze must be accessible locally

## Usage

### Basic Usage
```bash
python3 analyze_any_repo.py /path/to/your/repository
```

### Examples

#### Analyze the current directory
```bash
python3 analyze_any_repo.py .
```

#### Analyze a specific repository
```bash
python3 analyze_any_repo.py ~/projects/my-awesome-project
```

#### Analyze a subdirectory
```bash
python3 analyze_any_repo.py ./src/components
```

## What the Analysis Provides

When you run the script, it will:

1. **Build a comprehensive context** including:
   - Complete repository structure (tree view)
   - Language distribution and file statistics
   - Key configuration files (Makefile, go.mod, package.json, etc.)
   - Repository size and file counts

2. **Perform semantic search** for relevant code sections based on your query (if provided)

3. **Send the full context to Ollama** (gemma3:1b by default) which will:
   - Generate a comprehensive project description in markdown
   - Provide specific, actionable improvement suggestions
   - Answer questions about the codebase
   - Identify risks, technical debt, and optimization opportunities

4. **Save results** to:
   - `llm_prompt.txt` - The exact prompt sent to the LLM
   - `llm_analysis.txt` - The LLM's complete response

## Customization

### Change the Ollama Model
Edit the script to change the model:
```python
response = analyzer.query_ollama_with_full_context(prompt, model="mistral")  # or "codellama", etc.
```

### Add a Specific Query
Modify the script to analyze specific aspects:
```python
# Instead of general analysis, focus on security
prompt = analyzer.build_full_context_prompt(query="authentication and security")
```

## Workflow for Any Repository

1. **Navigate to the indexing directory** (where the scripts are located)
2. **Run the analysis script** pointing to your target repository:
   ```bash
   python3 analyze_any_repo.py /path/to/target/repository
   ```
3. **Wait for completion** (time depends on repository size and model speed)
4. **Review the results** in:
   - Console output (immediate feedback)
   - `llm_analysis.txt` (detailed analysis)
   - `llm_prompt.txt` (see exactly what was sent to the LLM)

## Example Output Analysis

The LLM will provide:
- **Executive Summary** - High-level project overview
- **Technical Architecture** - Layers, components, responsibilities
- **Key Components & Responsibilities** - Detailed breakdown
- **Dependencies & Integrations** - External libraries and services
- **Potential Risks & Challenges** - Areas of concern
- **Optimization Opportunities** - Performance and maintainability improvements
- **Deliverables & Next Steps** - Suggested development path

## Tips for Best Results

1. **Ensure Ollama is running**: Keep `ollama serve` running in a terminal
2. **Use appropriate models**: 
   - `gemma3:1b` - Good balance of speed and quality
   - `mistral` - Strong reasoning capabilities
   - `codellama` - Specifically trained for code understanding
3. **Large repositories**: The analysis may take longer for very large codebases
4. **Iterative analysis**: Run multiple times with different queries to explore specific aspects
5. **Combine with other tools**: Use the output to guide further investigation with grep, find, or IDE tools

## Troubleshooting

### "Vector store metadata not found"
- Run the full indexing pipeline first: `cd /path/to/repository && python3 /path/to/indexing/full_pipeline.py`
- This creates the vector store needed for semantic search

### Ollama connection errors
- Ensure Ollama is running: `ollama serve`
- Verify the model is available: `ollama list`
- Pull the model if needed: `ollama pull gemma3:1b`

### Slow performance
- The first run may be slower as models load into memory
- Subsequent analyses of the same repository will be faster
- Consider using smaller models for faster iteration

## Integration with Your Workflow

You can incorporate this analysis into:
- Code review processes
- Onboarding new developers
- Technical debt assessment
- Pre-refactoring assessment
- Architecture decision making
- Technical documentation generation

The analysis provides a solid foundation for understanding any codebase quickly and deeply.