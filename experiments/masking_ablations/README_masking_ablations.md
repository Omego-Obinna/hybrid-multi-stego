# Ours-QIM Masking Ablations

This package focuses only on Ours-QIM. It does not include Ours-XOR.

## What is evaluated

The mask-only experiment operationalises:

```text
eta = psi(o)
d   = PRF(k_e, eta)
mu  = QIM_Delta(P_params; d)
b   = m XOR mu
```

It varies:

- `psi(o)`: `hashid`, `global`, and `block`;
- QIM step: `0.5 Delta_0`, `Delta_0`, and `2 Delta_0`;
- dither source: cover-conditioned PRF, key-only PRF, and zero dither;
- message class: zero, alternating, low-entropy, text, and random.

No image embedding or STC execution is needed for the mask-only experiment.

## Important implementation warning

The existing image logs contain QIM cost-modulation fields such as
`qim_modulated_cost_*` and `qim_dither_alpha`. Those fields may describe the
image embedding cost rather than the formal payload mask. Run the audit first:

```bash
bash run_masking_ablations.sh audit
```

The package's `qim_mask()` is a concrete dithered-QIM instantiation of the
formal masking operator. If the current source code already implements a
different exact QIM mask readout, replace only that function.

## Install

```bash
mkdir -p /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations

unzip -o \
  /home/omego/Downloads/masking_ablations_experiment_package.zip \
  -d /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations

chmod +x \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/*.py \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/*.sh

source /home/omego/Documents/Hybrid_Stego_Eva/myenv/bin/activate

python -m pip install -r \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/requirements_masking_ablations.txt
```

## Benchmark key

Use a benchmark-only key, not a deployment secret:

```bash
export QIM_MASK_ABLATION_KEY="$(
python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
```

Keep the same key when repeating the paper run.

## Run order

```bash
bash \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/run_masking_ablations.sh \
  audit
```

```bash
bash \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/run_masking_ablations.sh \
  smoke
```

```bash
bash \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/run_masking_ablations.sh \
  paper
```

The paper run uses 50 covers, three payloads, three seeds, 11 QIM
configurations, and five message classes. At 0.10 bpp, each condition contains
more than one million bits per seed.

## NIST STS export

Select a configuration after reviewing
`paper/analysis/tables/qim_mask_ablation_summary.csv`:

```bash
export NIST_CONFIG_ID="cfg_01"
export NIST_PAYLOAD="0.10"
export NIST_MESSAGE_CLASS="zero"

bash \
  /home/omego/Documents/Hybrid_Stego_Eva/masking_ablations/run_masking_ablations.sh \
  nist
```

This produces ASCII sequences for the official NIST STS file-input workflow.
The NIST software itself is not bundled.

## Main outputs

```text
masking_ablations_outputs/paper/
├── generated/
│   ├── mask_sample_metrics.csv
│   ├── mask_classifier_features.csv
│   ├── mask_ablation_configs.csv
│   ├── bitstream_manifest.csv
│   └── bitstreams/
├── analysis/
│   ├── masking_ablations_draft.tex
│   ├── tables/
│   │   ├── qim_mask_ablation_summary.csv
│   │   ├── qim_mask_classifier_results.csv
│   │   ├── table_qim_mask_ablation.tex
│   │   └── table_qim_mask_classifier.tex
│   └── figures/
└── recovery/
    ├── qim_existing_recovery_summary.csv
    └── table_qim_existing_recovery.tex
```

## Interpretation

NIST-style tests, min-entropy estimates, and classifier results are empirical
diagnostics. They do not prove cryptographic pseudorandomness. The normalised
classifier proxy

```text
epsilon_param_proxy = 2 * abs(accuracy - 0.5)
```

is detector-dependent and is not a certified upper bound on total variation.
