# Feature Space Explorer

An interactive Streamlit application for teaching core feature-space ideas in machine learning through guided visual exploration.

## Overview

The project is designed for undergraduate learning and project demonstration. It combines:

- preprocessing and feature engineering workflows
- kernel trick visualisations
- PCA exploration and interpretation
- a grounded local AI tutor powered by Ollama

## Modules

- `Preprocessing`: inspect raw data, clean it, transform features, and export the processed result
- `Kernel Trick`: explore why linear models fail on nonlinear data and how kernels help
- `PCA Explorer`: study variance, loadings, scores, and reconstruction trade-offs
- `AI Tutor`: ask page-aware questions based on the learner's current view

## Tech Stack

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- Ollama for local LLM inference

## Project Structure

```text
app/
  components/   reusable Streamlit UI components
  pages/        module pages shown in the app
  main.py       Streamlit entry point
src/
  config/       app settings
  content/      instructional text and theory panels
  data/         built-in dataset loaders
  features/     preprocessing helpers
  kernel/       kernel-related utilities
  llm/          tutor client, context grounding, prompting
  pca/          PCA pipeline logic
  utils/        shared helpers
```

## Running Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Ollama if you want the AI tutor enabled:

```bash
ollama serve
ollama pull mistral
```

3. Run the app:

```bash
streamlit run app/main.py
```

## Notes

- The tutor is grounded to the current page state to reduce hallucinations.
- Sample datasets are included for quick demos.
- The app is intended for educational use and concept explanation rather than production ML pipelines.
