# Probing

## Layout

```
probing/
  run_probes_stem_final.py            # stem-final consonant identity probe
  run_probes_stemfinal_lnl.py         # L vs NL probe over stem-final representations
  run_probes_lnl_within_stemfinal.py  # L vs NL restricted to shared-stem-final subset
  summarize_lnl_within_stemfinal.py   # aggregate results into figures/tables
  run_lnl_within_stemfinal_probe_set.sh  # sequential wrapper over archs x runs
  extract_labels.py, control_tasks.py, utils/content_mask.py  # shared modules
  config.json                         # probe hyperparameters
data/
  probing/        # probe inputs/outputs: representations, labels, results_* (24 GB, gitignored)
```
## Running

```bash
python -m probing.run_probes_stem_final --help
bash probing/run_lnl_within_stemfinal_probe_set.sh
```
