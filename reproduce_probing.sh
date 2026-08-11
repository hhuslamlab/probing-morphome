#!/usr/bin/env bash
# =============================================================================
# Reproducible pipeline for the probing paper, one stage per Methodology
# subsection. Every stage is idempotent (per-output self-skip), so
# the script is safe to re-run and to resume after interruption.
#
#   bash reproduce_probing.sh all            # everything, in order
#   bash reproduce_probing.sh probes assets  # selected stages
#   ARCHS="vanilla" RUNS="1_2" bash reproduce_probing.sh probes
#
# Stages (-> paper section):
#   extract    §3.4  representation extraction        [drive + checkpoints]
#   pool       §3.5  content-mean pooling              [drive, SEQUENTIAL]
#   probes     §3.6-3.8 linear/MLP probes + controls   [local cache]
#   baselines  §3.11 surface n-gram + LM baselines     [text only]
#   within     §3.12 L/NL within stem-final subsets    [drive]
#   positional §3.14 stem-final-position readout       [drive for pooling]
#   transfer   §3.15 cross-subset transfer             [local cache]
#   nonce      §3.16 nonce-verb probing                [checkpoints]
#   structure  §3.13 cell clustering + conjugation ctrl [local cache]
#   summarize  §4    summary CSVs + trajectory plots
#   assets     §4    paper tables printout + figures
#   test       —     pytest suite for the analysis code
#
# Hardware notes: stages marked [drive] stream multi-GB tensors from
# REPS_DIR; run them sequentially and never in parallel with each other.
# CPU stages parallelise with PAR (default 4). BLAS is pinned per-worker.
#
# Float reproducibility: results are bit-deterministic at a FIXED
# BLAS_THREADS, but lbfgs trajectories differ in the last decimals across
# thread counts. Keep BLAS_THREADS constant across a sweep you intend to
# compare bitwise.
# =============================================================================
set -u
cd "$(dirname "$0")" || exit 1

FEATURE_INFORMED_ROOT=${FEATURE_INFORMED_ROOT:-$HOME/projects/research/feature_informed}
# Use the training repo's venv when present; otherwise stay in the caller's
# environment (pip install -r requirements.txt).
if [ -f "$FEATURE_INFORMED_ROOT/.venv/bin/activate" ]; then
  source "$FEATURE_INFORMED_ROOT/.venv/bin/activate"
fi

SPLIT=${SPLIT:-10L_90NL}
read -r -a ARCHS <<< "${ARCHS:-vanilla character_separated feature_invariant independent_feature feature_geometric}"
read -r -a RUNS  <<< "${RUNS:-1_1 1_2 1_3 1_4 2_1 2_2 2_3 2_4 3_1 3_2 3_3 3_4}"
REPS_DIR=${REPS_DIR:-}   # directory holding the extracted representations
PAR=${PAR:-4}
export OMP_NUM_THREADS=${BLAS_THREADS:-2} OPENBLAS_NUM_THREADS=${BLAS_THREADS:-2} MKL_NUM_THREADS=${BLAS_THREADS:-2}

log() { echo "[reproduce $(date +%H:%M:%S)] $*"; }

need_drive() {
  if [ -z "$REPS_DIR" ] || [ ! -d "$REPS_DIR" ]; then
    log "SKIP $1: representations not available (set REPS_DIR)"
    return 1
  fi
}

pairs() { for a in "${ARCHS[@]}"; do for r in "${RUNS[@]}"; do echo "$a $r"; done; done; }

# Run "module arch run [extra args...]" for every (arch, run), PAR-wide.
sweep() {
  local module=$1; shift
  pairs | xargs -n2 -P"$PAR" bash -c '
    python -u -m '"$module"' --model-type "$0" --split '"$SPLIT"' --run "$1" '"$*"'
    rc=$?; [ $rc -eq 0 ] || [ $rc -eq 2 ] || echo "FAILED: '"$module"' $0/$1 (exit $rc)"'
}

stage_extract() {  # §3.4 — see the per-architecture extractors for data layout
  need_drive extract || return
  for a in "${ARCHS[@]}"; do for r in "${RUNS[@]}"; do
    case "$a" in
      vanilla)
        python -u -m probing.extract_representations_vanilla \
          --model "${SPLIT}_$r" --checkpoint-name checkpoint_best.pt \
          --num-threads 4 --output-dir "$REPS_DIR" ;;
      character_separated)
        # run 1_1 has no checkpoint_best (never saved); it stays on _last.
        ck="checkpoint_best.pt"; [ "$r" = "1_1" ] && ck="checkpoint_last.pt"
        python -u -m probing.extract_representations_char_sep \
          --model-type character_separated \
          --checkpoint "$FEATURE_INFORMED_ROOT/checkpoints/char_sep/seperate_char_checkpoints/${SPLIT}_${r}-models/$ck" \
          --data-bin "$FEATURE_INFORMED_ROOT/data/char_sep_databin_aligned/${SPLIT}_$r" \
          --test-src "$FEATURE_INFORMED_ROOT/data/seperate_char_data/test.${SPLIT}_$r.src" \
          --test-tgt "$FEATURE_INFORMED_ROOT/data/seperate_char_data/test.${SPLIT}_$r.tgt" \
          --output-dir "$REPS_DIR" ;;
      independent_feature)
        python -u -m probing.extract_representations \
          --model-type "$a" --split "$SPLIT" --run "$r" \
          --checkpoint "$FEATURE_INFORMED_ROOT/checkpoints/feature_onehot/independentfeature_fixed/${SPLIT}_${r}.nll_0.0000.epoch_103" \
          --data-dir "$FEATURE_INFORMED_ROOT/data" --output-dir "$REPS_DIR" ;;
      *)
        python -u -m probing.extract_representations \
          --model-type "$a" --split "$SPLIT" --run "$r" \
          --checkpoint "$FEATURE_INFORMED_ROOT/checkpoints/$a/${SPLIT}_$r" \
          --data-dir "$FEATURE_INFORMED_ROOT/data" --output-dir "$REPS_DIR" ;;
    esac
  done; done
}

stage_pool() {  # §3.5 — sequential: concurrent multi-GB reads thrash the drive
  need_drive pool || return
  for a in "${ARCHS[@]}"; do for r in "${RUNS[@]}"; do
    python -u -m probing.pool_representations --model-type "$a" \
      --split "$SPLIT" --run "$r" --representations-dir "$REPS_DIR" \
      --cache-dir data/probing/pooled_cache || return 1
  done; done
}

stage_probes() {  # §3.6-3.8 — balanced accuracy, lemma-disjoint folds, controls
  sweep probing.run_probes_stemfinal_lnl \
    --data-dir "$FEATURE_INFORMED_ROOT/data" \
    --pooled-cache-dir data/probing/pooled_cache --n-jobs 2 \
    --control --n-controls 5
}

stage_baselines() {  # §3.11 — classifier baselines depend only on the lemma
  # split, so one run per split suffices; the LM refits per fold regardless.
  for r in 1_1 2_2 3_2; do
    python -u -m probing.run_ngram_baselines --split "$SPLIT" --run "$r" \
      --data-dir "$FEATURE_INFORMED_ROOT/data" \
      --output-dir data/probing/results_ngram_baselines_balanced --n-jobs 4
  done
}

stage_within() {  # §3.12 — reads representations directly (drive)
  need_drive within || return
  sweep probing.run_probes_lnl_within_stemfinal \
    --representations-dir "$REPS_DIR" --n-jobs 1
}

stage_positional() {  # §3.14 — pooling pass is sequential (drive), probes local
  need_drive positional || return
  for a in "${ARCHS[@]}"; do for r in "${RUNS[@]}"; do
    python -u -m probing.pool_stemfinal_position --model-type "$a" \
      --split "$SPLIT" --run "$r" --representations-dir "$REPS_DIR"
    rc=$?; [ $rc -eq 0 ] || [ $rc -eq 2 ] || return 1
  done; done
  sweep probing.run_probes_positional
  sweep probing.run_probes_positional --suffix prealt
}

stage_transfer() {  # §3.15 — local pooled cache only
  sweep probing.run_transfer_probe
}

stage_nonce() {  # §3.16 — re-extracts the 120 wug items per model (checkpoints)
  python -u -m probing.run_nonce_probe --prepare
  sweep probing.run_nonce_probe
}

stage_structure() {  # §3.13 morphome-structure probes (cell clustering + conjugation control)
  sweep probing.run_morphome_structure
}

stage_summarize() {  # §4 — summary CSVs; NOTE their accuracy_mean is RAW
  python -m probing.summarize_stemfinal_lnl
  python -m probing.summarize_lnl_within_stemfinal
}

stage_assets() {  # §4 — paper tables (balanced, from per-run CSVs) + figures
  python -m probing.make_paper_assets
}

stage_test() {
  python -m pytest tests/ -q
}

STAGES=${*:-all}
[ "$STAGES" = "all" ] && STAGES="extract pool probes baselines within positional transfer nonce structure summarize assets test"
for s in $STAGES; do
  log "=== stage: $s ==="
  "stage_$s" || log "stage $s reported an error"
done
log "done"
