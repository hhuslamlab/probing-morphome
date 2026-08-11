"""docopt CLI regression tests: docopt 0.6.2 treats any docstring line whose
first non-space character is a dash as an option declaration, so a stray prose
line silently breaks the CLI; these parses lock each module's CLI in place."""

import pytest
from docopt import docopt

import probing.extract_labels as labels
import probing.extract_representations as extract
import probing.extract_representations_char_sep as extract_cs
import probing.extract_representations_vanilla as extract_v
import probing.make_paper_assets as assets
import probing.pool_representations as pool
import probing.pool_stemfinal_position as posp
import probing.run_ngram_baselines as ngram
import probing.run_probes_lnl_within_stemfinal as within
import probing.run_probes_positional as pospr
import probing.run_probes_stemfinal_lnl as probes
import probing.run_transfer_probe as transfer
import probing.summarize_lnl_within_stemfinal as sum_within
import probing.summarize_stemfinal_lnl as sum_probes


STANDARD = ["--model-type", "vanilla", "--split", "10L_90NL", "--run", "1_2"]


class TestDocstringsParse:
    def test_positional_pooling(self):
        a = docopt(posp.__doc__, argv=STANDARD + ["--representations-dir", "/x"])
        assert a["--representations-dir"] == "/x"
        assert a["--cache-dir"] == "data/probing/pooled_cache"

    def test_positional_probes(self):
        a = docopt(pospr.__doc__, argv=STANDARD)
        assert a["--probe-types"] == "linear"

    def test_transfer(self):
        a = docopt(transfer.__doc__, argv=STANDARD)
        assert a["--n-controls"] == "5"
        assert a["--model-type"] == "vanilla"
        assert a["--seed"] == "42"  # default from Options block

    def test_assets(self):
        a = docopt(assets.__doc__, argv=["--no-figures"])
        assert a["--no-figures"]

    def test_extract(self):
        a = docopt(extract.__doc__, argv=STANDARD + ["--checkpoint", "/c.pt"])
        assert a["--baseline"] == "none"
        assert a["--baseline-seed"] == "1337"

    def test_extract_vanilla_all_defaults(self):
        a = docopt(extract_v.__doc__, argv=[])
        assert a["--checkpoint-name"] == "checkpoint_best.pt"
        assert a["--min-token-acc"] == "0.5"
        assert a["--fairseq-data-root"] is None

    def test_extract_char_sep_two_modes(self):
        a = docopt(extract_cs.__doc__, argv=["--checkpoint", "/c.pt", "--data-bin", "/db"])
        assert not a["--all"] and a["--model-type"] == "char_sep"
        b = docopt(extract_cs.__doc__, argv=["--all", "--data-root", "/dr"])
        assert b["--all"] and b["--checkpoint"] is None

    def test_labels(self):
        a = docopt(labels.__doc__, argv=["--split", "10L_90NL", "--run", "1_1"])
        assert a["--control-salt"] == "phase1-control-v1"
        assert not a["--control-task"]

    def test_pool(self):
        a = docopt(pool.__doc__, argv=STANDARD)
        assert a["--pool-positions"] == "content"
        assert a["--chunk-size"] == "8192"

    def test_probes(self):
        a = docopt(probes.__doc__, argv=STANDARD + ["--control", "--n-jobs", "2"])
        assert a["--cv-mode"] == "grouped"
        assert a["--probe-types"] == "linear"
        assert a["--control"] and a["--pooled-cache-dir"] is None

    def test_within(self):
        a = docopt(within.__doc__, argv=STANDARD)
        assert a["--n-controls"] == "5"
        assert a["--output-dir"] == "data/probing/results_lnl_within_stemfinal"

    def test_ngram(self):
        a = docopt(ngram.__doc__, argv=["--split", "10L_90NL", "--run", "2_2"])
        assert a["--ngram-orders"] == "1 2 3"
        assert a["--baselines"] == "classifier lm"
        assert a["--output-dir"] == "data/probing/results_ngram_baselines_balanced"

    def test_summarizers(self):
        a = docopt(sum_probes.__doc__, argv=[])
        assert a["--results-dir"] == "data/probing/results_stemfinal_lnl_grouped"
        assert a["--baselines-dir"] == "data/probing/results_ngram_baselines_balanced"
        b = docopt(sum_within.__doc__, argv=["--probe-type", "mlp"])
        assert b["--probe-type"] == "mlp"


class TestCliHelper:
    def test_choices_rejection(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["x", "--model-type", "bogus", "--split", "10L_90NL",
                             "--run", "1_2"])
        with pytest.raises(SystemExit):
            transfer.parse_args()

    def test_full_parse_maps_attributes(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv", ["x", *STANDARD, "--n-controls", "7"])
        args = transfer.parse_args()
        assert args.model_type == "vanilla"
        assert args.n_controls == 7
        assert args.output_dir == "data/probing/results_transfer"
        assert args.data_dir.endswith("/data")  # FEATURE_INFORMED_DATA resolved

    def test_ngram_list_options_are_split_and_validated(self, monkeypatch):
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["x", "--split", "10L_90NL", "--run", "1_1",
                             "--ngram-orders", "1 2", "--baselines", "lm"])
        args = ngram.parse_args()
        assert args.ngram_orders == [1, 2]
        assert args.baselines == ["lm"]
        monkeypatch.setattr(sys, "argv",
                            ["x", "--split", "10L_90NL", "--run", "1_1",
                             "--baselines", "bogus"])
        with pytest.raises(SystemExit):
            ngram.parse_args()
