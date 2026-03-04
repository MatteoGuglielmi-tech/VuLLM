# VuLLM

Fine-tuning Large Language Models for C Vulnerability Detection and Classification (CWE) with Structured Reasoning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and experimental pipeline for my master's thesis on using fine-tuned LLMs for automated vulnerability detection and classification in C code. The approach generates structured JSON outputs containing both natural language security reasoning and CWE (Common Weakness Enumeration) classifications.

**Key findings:**
- Pessimistic assumptions combined with CWE guidance achieve 4.3× higher F1-score than Random Forest baseline
- Neither assumptions alone nor CWE guidance alone achieves substantial improvement—only their combination is effective
- Training data quality (not prompt design) is the primary bottleneck, with a 15.5% recall ceiling across all configurations
- Diagnostic suite validation shows 91.7% accuracy on handcrafted test cases

## Repository Structure
```sh
VuLLM/
├── .clang-format               # Clang-format config file
├── deepspeed                   # Deepspeed files
├── diagnostics                 # 24-case diagnostic suite
│   ├── assets                  # Results diagnostic suite
│   ├── none                    # Predictions Free (no assumptions)
│   ├── optimistic              # Predictions Optimistic assumptions
│   ├── pessimistic             # Predictions Pessimistic assumptions
│   └── yaml_parser             # Results renderer
├── DoneBot                     # Submodule for async notifications
├── LICENSE
├── pixi.toml                   # Dependency management
├── pyproject.toml
├── README.md
├── rusty                       # Rust implementation
│   ├── Cargo.toml
│   ├── pixi.toml
│   ├── tests
│   ├── src
│   │   ├── lib.rs
│   │   ├── main.rs
│   │   ├── mitre               # MITRE db entries
│   │   └── processor_lib       # Tree-sitter parsing, GCC repair, AST validation, CWE enrichment
├── src                         # Python
│   ├── core
│   │   ├── cot                 # CoT generation and Jury for quality assessment
│   │   ├── cot_training        # Fine-tuning
│   │   └── random_forest       # RFC baseline
│   ├── dataset                 # Dataset utilities
│   └── test_env_integrity      # Environment validation
└── text_prompts                # Prompts applied
```

## Requirements

- **Preprocessing:** Rust 1.89+
- **Training/Evaluation:** Python 3.12
- **Hardware:** NVIDIA L40s GPU with 48GB VRAM for training
- **Dependencies:** Managed via [pixi](https://pixi.sh)

## Installation
```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/MatteoGuglielmi-tech/VuLLM.git
cd VuLLM

# If already cloned without submodules, initialize them:
# git submodule update --init --recursive

# Install dependencies
pixi install

# Build Rust preprocessing pipeline
cd rusty && cargo build --release
```

## Usage

Each component includes built-in argument parsing. Use `--help` for available options and usage examples.

| Component | Command |
|-----------|---------|
| Preprocessing | `cd rusty && cargo run --release -- --help` |
| Training/Evaluation | `pixi run python python -m src.core.cot_training.main --help` |


## Experimental Configurations

The thesis evaluates 6 configurations in a 3×2 factorial design:

| Config | Assumption Mode | CWE Guidance | F1-Score | Recall |
|--------|-----------------|--------------|----------|--------|
| 1 | Free | No | 15.1% | 8.7% |
| 2 | Free | Yes | 15.9% | 9.1% |
| 3 | Optimistic | No | 12.9% | 7.2% |
| 4 | Optimistic | Yes | 17.3% | 9.8% |
| 5 | Pessimistic | No | 15.2% | 8.7% |
| **6** | **Pessimistic** | **Yes** | **24.8%** | **15.5%** |

## Dataset

This work uses the [DiverseVul](https://github.com/wagner-group/diversevul) dataset. Due to licensing, we cannot redistribute processed data. The preprocessing pipeline can be applied to the publicly available dataset to reproduce our results.

**Final dataset:** 5888 samples (4302 train / 743 val / 843 test)

<!-- ## Citation -->
<!---->
<!-- If you use this code in your research, please cite: -->
<!-- ```bibtex -->
<!-- @mastersthesis{guglielmi2025vullm, -->
<!--     author = {Guglielmi, Matteo}, -->
<!--     title = {Fine-tuning Large Language Models for C Vulnerability Detection with Structured Reasoning}, -->
<!--     school = {University of Trento}, -->
<!--     year = {2025}, -->
<!--     type = {Master's Thesis} -->
<!-- } -->
<!-- ``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DiverseVul dataset authors for making vulnerability data publicly available
- Qwen team for the Qwen2.5-Coder model
- Unsloth for efficient fine-tuning infrastructure
