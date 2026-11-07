# Multilingual text samples

Upstream Strata's README translations (German, Spanish, French, Japanese, Polish, Brazilian Portuguese, Simplified
Chinese), as of 29 Oct 2026. They describe Strata, not Xolotl, so they moved out of the repository root on 8 Nov 2026.
They stay because calibration and expert-pruning tools read them as non-English text: `tools/v41/v41_calib.py` (the
European and Chinese calibration sets) and `tools/ds4/make_reap_prompts.py`.
