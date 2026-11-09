# DONER

Domain-Informed Operator-Guided Noise Encoding and Restoration denoises a single grayscale image. This release includes the final STM, infrared and confocal models, their frozen training priors, 300 reference images, five comparison model implementations and the final numerical results. Run the commands below from the directory containing this README.

## Environment

The reference training environment used Linux, Python 3.12.3, PyTorch 2.5.1 with CUDA 12.4, NumPy 2.1.3, SciPy 1.14.1 and Pillow 11.0.0. The full-resolution infrared training configuration uses two CUDA devices and NCCL. STM and confocal training each use one device. Inference also runs on CPU. The release was checked on CPU with Python 3.12 and PyTorch 2.6.0.

Create a Python 3.12 environment. For a Linux CUDA 12.4 installation, use the following commands.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.1.3 scipy==1.14.1 pillow==11.0.0 einops
```

For CPU inference on macOS, install PyTorch with `python -m pip install torch==2.6.0` in place of the CUDA installation line. The optional loader for the official IDF checkpoint also uses `omegaconf`.

## Run the supplied models

This example adds Gaussian noise to one included STM reference and restores it with the released weights. It illustrates the interface with a new input. The paper's scores are in `results/`.

```bash
mkdir -p runs/example
python - <<'PY'
from pathlib import Path
import numpy as np
clean = np.load(Path("data/stm/validation/00004.npy"), allow_pickle=False)
noise = np.random.default_rng(42).normal(0, 0.05, clean.shape).astype(np.float32)
np.save("runs/example/noisy.npy", clean + noise, allow_pickle=False)
PY
python -m doner.infer --domain stm --checkpoint weights/stm.pt --input runs/example/noisy.npy --output runs/example/restored.npy --device cpu
```

For your own observations, use the matching domain and checkpoint. Set `--device cuda` for GPU inference.

| Domain | Inference checkpoint | Training prior | Experiment image size (H × W) |
|---|---|---|---|
| `stm` | `weights/stm.pt` | `weights/priors/stm.pt` | 300 × 300 |
| `ir` | `weights/ir.pt` | `weights/priors/ir.pt` | 768 × 1024 |
| `confocal` | `weights/confocal.pt` | `weights/priors/stm.pt` | 128 × 128 |

Each inference checkpoint contains the complete 5,623,796-parameter denoiser, including its frozen component. It was exported from the corresponding 21,000-update endpoint. The standalone priors initialize new training runs. Supply one finite, two-dimensional observation with at least 32 pixels on each side. Floating-point NPY inputs use the experiment's normalized intensity scale. For 8-bit PNG/TIFF input, the loader divides intensity by 255. It also accepts RGB files whose channels are identical. The output is an unclipped float32 NPY array in the input geometry.

## Train

The packaged operator libraries under `doner/operators/` are ready to use. Each run takes its 50 training references from `data/index.csv` in the original sampling order. Choose a fresh output directory for each invocation.

```bash
python -m doner.train --domain stm --data data/stm/train --prior weights/priors/stm.pt --output runs/stm
python -m doner.train --domain confocal --data data/confocal/train --prior weights/priors/stm.pt --output runs/confocal
torchrun --standalone --nproc_per_node=2 -m doner.train --domain ir --data data/ir/train --prior weights/priors/ir.pt --output runs/ir
```

The default schedule writes `final.pt` after 21,000 updates. For a short installation check, append `--steps 1` to the confocal command and choose a new output directory. This produces `latest.pt`. The confocal sampler and losses change at steps 2,049 and 6,157 as part of the final training schedule. New runs may differ numerically across library versions and devices.

## Reference data and numerical results

Each domain has 50 training and 50 fixed validation references. Paths, source identifiers, split order, image dimensions, and file and float32 pixel hashes are in `data/index.csv`.

| Domain | Reference image | Representation |
|---|---|---|
| STM | Simulated MoS₂ height maps, normalized per map | Float32 NPY |
| Infrared | M3FD high-quality infrared frames used to synthesize noisy inputs | Original lossless 8-bit PNG |
| Confocal | Hagen actin and mitochondrial higher-SNR acquisitions, with 25 fields of view per split | Float32 NPY crops, already normalized by 65,535 |

The infrared and confocal references are captured images and may retain acquisition noise. Hagen dataset accession [10.5524/100888](https://doi.org/10.5524/100888).

`results/manuscript_comparison.csv` gives the published PSNR and SSIM values. `results/case_scores.csv` gives 18,300 method-case records for six methods on 3,050 observations. STM and infrared each contribute 1,200 synthetic observations. Confocal contributes 600 synthetic observations and 50 measured pairs evaluated against the corresponding higher-SNR targets. The main confocal synthetic table uses the mixed-grain family. The remaining CSV files contain per-family means, DONER's PSNR difference from the strongest baseline, the reported ablations, and parameter counts. These are fixed validation results. The release includes the reference side of the data only.

## Full validation replay

The three final checkpoints were evaluated on all 3,050 scored observations. The 24 PSNR and SSIM entries in the main tables match the manuscript at its displayed precision. All 2,400 STM and infrared predictions match the archived outputs byte for byte. On the 650 confocal observations, CPU inference differed from the archived CUDA scores by at most 0.000014 dB in PSNR and 0.00000060 in SSIM per image. The [table comparison](validation/main_table_comparison.csv), [per-image differences](validation/case_differences.csv) and [execution record](validation/verification.json) contain the full results.

## Comparison models

`baselines/` contains the inference architectures and grayscale adapters for DEAL, RNINet, CLIPDenoising, IDF and CENSURE. Supply the appropriate external baseline checkpoint. DEAL accepts a training checkpoint containing `models.NE` or a plain NE state dictionary. RNINet and CLIPDenoising accept a full trained network in `model` or as a plain state dictionary. IDF uses the official `idf_g_15.ckpt` with its stored normalization. CENSURE uses the official grayscale state dictionary. Example with an IDF checkpoint saved under `external/`:

```bash
python -m baselines.infer --method IDF --checkpoint external/idf_g_15.ckpt --input runs/example/noisy.npy --output runs/example/idf.npy --device cpu
```

The upstream implementations used for the comparison networks are [DEAL](https://github.com/LiuZhu-CV/DEAL/tree/472764e6cdfca5b2e937f6bc7ae50eafc641f0d9), [RNINet](https://github.com/default-8541/RNINet/tree/4394e8391b2c0fc2ca482853b074617875d8c47e), [CLIPDenoising](https://github.com/alwaysuu/CLIPDenoising/tree/2647bf815df0299a03d16df34b896fa266ceefda), [IDF](https://github.com/dongjinkim9/IDF/tree/bf088e1eba2caee472612eee6392dc92d37a3d8e) and [CENSURE](https://github.com/sherbret/divergence_free_nn/tree/6b519686c7b35ee310ca55568cb8d956eb742a0c). The release uses local imports and a common inference entry point. CLIPDenoising's encoder can be restored directly from its trained state. Its encoder utilities credit Zongsheng Yue and [pytorch-unet](https://github.com/jvanvugt/pytorch-unet). The lightweight DRUNet used by CENSURE credits [DPIR](https://github.com/cszn/DPIR). Upstream license notices are in `licenses/`.

## Optional offline texture generation

The denoiser and its training commands use the packaged fitted libraries. To regenerate textures, use a CUDA device, Diffusers 0.35.1, Transformers 4.49.0, Accelerate, Safetensors and Hugging Face Hub. Prepare the FP16 components of [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b) at the revision pinned in `doner/offline/pinned_model.json`. Place that JSON in the downloaded model directory alongside the component files it lists. Set `SDXL_DIR` to this directory.

```bash
python -m pip install diffusers==0.35.1 transformers==4.49.0 accelerate safetensors huggingface_hub
python -m doner.offline.guided_bank --root runs/offline --model-path "$SDXL_DIR" --profiles doner/offline/guided_profiles.json --domain all
python -m doner.offline.fit_generated --root runs/offline
```

The fitted infrared and STM libraries are written under `runs/offline/operators/`. For confocal textures, use the included prompt protocol:

```bash
python -m doner.offline.confocal_bank --root runs/confocal_bank --protocol doner/operators/confocal/prompt_protocol.json --model-path "$SDXL_DIR"
python -m doner.offline.confocal_bank --root runs/confocal_bank --protocol doner/operators/confocal/prompt_protocol.json --model-path "$SDXL_DIR" --text-probe-only
```

Review the generated candidates and record the accepted ones in `runs/confocal_bank/admission.json` using the generated `admission_template.json`. Fit the new library with:

```bash
python -m doner.offline.confocal_bank --root runs/confocal_bank --protocol doner/operators/confocal/prompt_protocol.json --fit-only
```
