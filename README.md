# Probing Character-level Transformers for the Spanish L-shaped Morphome

Code and result data for the probing experiments in [*Probing Character-level
Transformers for the Spanish L-shaped Morphome*](https://arxiv.org/abs/2608.03452).

## Layout

```
probing/
  extract_representations.py           
  extract_representations_vanilla.py   
  extract_representations_char_sep.py  
  extract_labels.py                    # morphological property labels from .src/.tgt test files
  pool_representations.py              # content-mean pooling
  pool_stemfinal_position.py           # stem-final position pooling (positional readout)
  run_probes_stemfinal_lnl.py          # main probes: stem_final_match / conjugation / l_shaped
  run_probes_lnl_within_stemfinal.py   # L vs NL within stem-final subsets
  run_probes_positional.py             # stem-final-position readout probes
  run_ngram_baselines.py               # surface n-gram + class-conditional LM baselines
  run_transfer_probe.py                # cross-subset transfer probes
  run_morphome_structure.py            # cell clustering + conjugation control
  summarize_stemfinal_lnl.py           # summary CSVs + trajectory plots
  summarize_lnl_within_stemfinal.py    # summary for the within-stem-final probes
  make_paper_assets.py                 # paper tables and figures
  control_tasks.py, analysis_common.py, utils/, config.json
tests/                                 # pytest suite for the analysis code
data/probing/results_*/                # result CSVs behind the paper's numbers 
reproduce_probing.sh                   # stage-by-stage reproduction 
```

## Installation

Python 3.11:

```bash
pip install -r requirements.txt
```

All modules are run as a package from the repo root:

```bash
python -m probing.run_probes_stemfinal_lnl --help
```

## Reproduction

`reproduce_probing.sh` is the entry point; each stage maps to a
subsection of the paper and is idempotent (safe to re-run and resume):

```bash
bash reproduce_probing.sh all            # everything, in order
bash reproduce_probing.sh probes assets  # selected stages
```

The `summarize`, `assets`, and `test` stages run out of the box from the
tracked result CSVs. The extraction stages additionally need the trained
checkpoints and the training/test data, resolved via `FEATURE_INFORMED_ROOT`
(training repo root) and `REPS_DIR` (extracted-representations directory).

## Checkpoints

All checkpoints used in the paper (5 architectures × 12 runs, 10L_90NL) are on
Hugging Face at
[akki2825/probing-morphome-checkpoints](https://huggingface.co/akki2825/probing-morphome-checkpoints),
mirroring the layout the pipeline expects:

```bash
hf download akki2825/probing-morphome-checkpoints --local-dir "$FEATURE_INFORMED_ROOT"
```

