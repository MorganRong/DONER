# DONER

Domain-Informed Operator-Guided Noise Encoding and Restoration denoises a single grayscale image. This release includes the final STM, infrared and confocal models, their frozen training priors, 300 reference images, five comparison model implementations and the final numerical results. Run the commands below from the directory containing this README.

[Main-paper results](#main-paper-results) · [Appendix results](#appendix-results) · [Validation replay](#full-validation-replay) · [Environment](#environment) · [Run the models](#run-the-supplied-models) · [Train](#train)

## Main-paper results

DONER exceeds the strongest baseline on mean PSNR by **7.25 dB on simulated STM**, **6.90 dB on infrared**, and **1.27 dB on measured confocal pairs**. Cells below give **PSNR (dB) / SSIM**. Both metrics are averaged over individual observations; higher is better. The gains use the unrounded means.

| Evaluation | Cases | DONER | CLIPDenoising | PSNR gain (dB) |
| --- | --- | --- | --- | --- |
| Simulated STM · all strengths | 1,200 | 37.97 / 0.9775 | 30.71 / 0.9047 | +7.25 |
| Infrared · all strengths | 1,200 | 32.06 / 0.8947 | 25.16 / 0.6168 | +6.90 |
| Confocal · measured pairs | 50 | 25.17 / 0.6756 | 23.90 / 0.6427 | +1.27 |

The fixed validation set contains 3,050 scored observations. All six methods receive the same input within a case. The [main-table scores](results/manuscript_comparison.csv) and [18,300 method–case records](results/case_scores.csv) give the underlying results.

### Simulated STM and infrared (Tables 1–2)

Each domain uses 50 validation references × 8 noise families × 3 strengths: 400 observations at each strength and 1,200 overall. STM uses simulated MoS₂ height maps; infrared uses native M³FD frames. **Bold** marks the best pair in each column.

**Table 1. Simulated STM**

| Method | a = 0.05 | a = 0.10 | a = 0.20 | All strengths |
| --- | --- | --- | --- | --- |
| DEAL | 22.30 / 0.6414 | 18.75 / 0.4375 | 13.64 / 0.2732 | 18.23 / 0.4507 |
| RNINet | 37.56 / 0.9803 | 29.37 / 0.9166 | 19.57 / 0.6614 | 28.83 / 0.8528 |
| CLIPDenoising | 36.85 / 0.9775 | 30.77 / 0.9268 | 24.52 / 0.8097 | 30.71 / 0.9047 |
| IDF | 26.87 / 0.7335 | 20.54 / 0.5207 | 14.40 / 0.3498 | 20.61 / 0.5346 |
| CENSURE | 27.76 / 0.7484 | 22.21 / 0.5734 | 17.06 / 0.4184 | 22.34 / 0.5801 |
| DONER | **42.57 / 0.9923** | **38.20 / 0.9836** | **33.13 / 0.9565** | **37.97 / 0.9775** |

**Table 2. Infrared**

| Method | a = 0.05 | a = 0.10 | a = 0.20 | All strengths |
| --- | --- | --- | --- | --- |
| DEAL | 19.84 / 0.5279 | 16.53 / 0.3809 | 12.29 / 0.2523 | 16.22 / 0.3870 |
| RNINet | 30.53 / 0.7997 | 23.44 / 0.5560 | 16.35 / 0.2982 | 23.44 / 0.5513 |
| CLIPDenoising | 31.22 / 0.8285 | 25.13 / 0.6255 | 19.13 / 0.3964 | 25.16 / 0.6168 |
| IDF | 26.82 / 0.6481 | 20.52 / 0.4024 | 14.42 / 0.2383 | 20.59 / 0.4296 |
| CENSURE | 28.22 / 0.6926 | 22.71 / 0.4849 | 17.47 / 0.2944 | 22.80 / 0.4906 |
| DONER | **36.04 / 0.9482** | **32.19 / 0.9068** | **27.94 / 0.8290** | **32.06 / 0.8947** |

### Confocal fluorescence (Tables 3–4)

The first three columns use synthetic **mixed grain only**, with 50 higher-SNR reference patches at each strength. The last column compares 50 measured low-SNR inputs with higher-SNR acquisitions of the same fields and channels. At the weakest synthetic strength, CLIPDenoising ranks first on both metrics; DONER ranks first at the two higher strengths and on the measured pairs.

| Method | a = 0.05 | a = 0.10 | a = 0.20 | Measured pairs |
| --- | --- | --- | --- | --- |
| DEAL | 18.83 / 0.4321 | 15.79 / 0.2067 | 12.37 / 0.0691 | 17.79 / 0.2723 |
| RNINet | 29.13 / 0.7759 | 24.89 / 0.6628 | 17.56 / 0.3421 | 23.82 / 0.6258 |
| CLIPDenoising | **30.62 / 0.8547** | 25.33 / 0.7074 | 19.40 / 0.4451 | 23.90 / 0.6427 |
| IDF | 27.57 / 0.7575 | 21.17 / 0.4748 | 15.00 / 0.2226 | 21.98 / 0.5386 |
| CENSURE | 28.38 / 0.7337 | 25.11 / 0.6155 | 21.19 / 0.4413 | 23.51 / 0.5992 |
| DONER | 30.39 / 0.8519 | **28.01 / 0.7665** | **26.11 / 0.6801** | **25.17 / 0.6756** |

### Results by noise family (Table 5)

Each cell is DONER’s mean PSNR minus the strongest of the five baselines on the same 50 references, in dB. Parentheses identify that baseline: **R** = RNINet, **C** = CLIPDenoising, **D** = DEAL, **S** = CENSURE. Negative values favor the baseline. Absolute PSNR and SSIM for every method are in the [per-family scores](results/noise_family_scores.csv); the [margin data](results/noise_family_advantage.csv) give the comparator by name.

| Domain | Noise family | a = 0.05 | a = 0.10 | a = 0.20 |
| --- | --- | --- | --- | --- |
| Simulated STM | Fine grain | -0.63 (R) | +0.85 (R) | +3.52 (C) |
| Simulated STM | Correlated grain | -0.72 (R) | +3.45 (R) | +4.64 (C) |
| Simulated STM | Thin scan-line stripes | +1.58 (R) | +4.83 (C) | +6.36 (C) |
| Simulated STM | Variable scan-line stripes | +2.86 (R) | +5.91 (C) | +7.55 (C) |
| Simulated STM | Sparse short streaks | +10.41 (C) | +11.46 (C) | +12.27 (C) |
| Simulated STM | Mixed short streaks | +10.37 (C) | +13.26 (C) | +14.35 (C) |
| Simulated STM | Grain + scan-line stripes | +2.56 (R) | +6.29 (C) | +7.99 (C) |
| Simulated STM | Grain + short streaks | +9.85 (C) | +11.74 (C) | +12.20 (C) |
| Infrared | Fine grain | -0.83 (C) | +1.11 (C) | +4.80 (C) |
| Infrared | Correlated grain | +2.52 (R) | +6.66 (C) | +9.30 (C) |
| Infrared | Heterogeneous grain | -0.24 (C) | +2.01 (C) | +4.88 (C) |
| Infrared | Variable column offsets | +8.88 (C) | +11.43 (C) | +11.02 (D) |
| Infrared | Intermittent column offsets | +9.26 (C) | +10.51 (C) | +10.06 (S) |
| Infrared | Variable row offsets | +5.63 (C) | +6.84 (C) | +6.77 (C) |
| Infrared | Smooth offset field | +4.56 (C) | +6.75 (C) | +9.02 (C) |
| Infrared | Grain + column offsets | +8.68 (C) | +11.19 (C) | +10.53 (D) |
| Confocal | Fine grain | -0.25 (C) | +2.01 (C) | +3.91 (S) |
| Confocal | Short-correlated grain | -0.27 (C) | +2.97 (C) | +5.94 (S) |
| Confocal | Low-count background | -0.17 (C) | +3.23 (R) | +5.84 (S) |
| Confocal | Mixed grain | -0.23 (C) | +2.67 (C) | +4.91 (S) |

DONER leads in 12 of 20 families at the weakest strength and in all 20 at each higher strength. At a = 0.20, its largest margins within STM, infrared and confocal are +14.35 dB for mixed short streaks, +11.02 dB for variable column offsets and +5.94 dB for short-correlated grain, respectively.

### Examples from the main paper

Figures 3 and 5 show grain mixed with horizontal STM streaks or with infrared column offsets. Figure 6 uses confocal field 022: the same mitochondrial patch at three synthetic strengths, plus measured actin and mitochondrial patches. The table below reports the full-frame STM and infrared scores and the 128 × 128 confocal patch scores. “Input” is the unrestored observation; “mito” denotes mitochondria.

<details>
<summary>Per-image scores for the 17 examples in Figures 3, 5 and 6</summary>

| Figure · reference · condition | Input | DEAL | RNINet | CLIPDenoising | IDF | CENSURE | DONER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fig. 3 · 00179 · 0.05 | 25.79 / 0.8271 | 22.32 / 0.7700 | 33.51 / 0.9599 | 34.54 / 0.9695 | 26.25 / 0.9079 | 26.37 / 0.9095 | 45.84 / 0.9886 |
| Fig. 3 · 00066 · 0.05 | 26.01 / 0.7790 | 22.12 / 0.7293 | 34.30 / 0.9603 | 35.06 / 0.9715 | 26.65 / 0.9124 | 26.96 / 0.9181 | 44.95 / 0.9883 |
| Fig. 3 · 00086 · 0.10 | 22.46 / 0.5934 | 20.40 / 0.5836 | 28.15 / 0.9536 | 32.28 / 0.9603 | 23.18 / 0.8461 | 24.67 / 0.9393 | 43.36 / 0.9824 |
| Fig. 3 · 00199 · 0.10 | 20.26 / 0.5443 | 18.73 / 0.5146 | 25.31 / 0.9148 | 29.57 / 0.9321 | 20.80 / 0.7680 | 22.67 / 0.9013 | 37.87 / 0.9744 |
| Fig. 3 · 00161 · 0.20 | 14.77 / 0.4124 | 14.13 / 0.3926 | 17.27 / 0.8045 | 22.99 / 0.8650 | 15.14 / 0.5982 | 17.94 / 0.8448 | 34.68 / 0.9611 |
| Fig. 3 · 00058 · 0.20 | 14.62 / 0.3505 | 14.14 / 0.3391 | 16.82 / 0.8083 | 23.16 / 0.8631 | 15.02 / 0.5001 | 18.75 / 0.8379 | 36.35 / 0.9493 |
| Fig. 5 · 03765 · 0.05 | 26.12 / 0.5504 | 21.54 / 0.7436 | 27.97 / 0.6676 | 28.63 / 0.7496 | 26.55 / 0.5847 | 26.22 / 0.5503 | 37.99 / 0.9724 |
| Fig. 5 · 02344 · 0.05 | 25.71 / 0.4589 | 23.08 / 0.6878 | 27.83 / 0.6055 | 28.35 / 0.6926 | 26.33 / 0.5208 | 25.97 / 0.4654 | 39.25 / 0.9547 |
| Fig. 5 · 03928 · 0.10 | 20.09 / 0.3024 | 20.50 / 0.5505 | 21.20 / 0.3860 | 21.34 / 0.4162 | 20.37 / 0.3246 | 20.43 / 0.3116 | 33.11 / 0.9460 |
| Fig. 5 · 01706 · 0.10 | 19.76 / 0.2279 | 21.85 / 0.5422 | 21.67 / 0.3365 | 22.18 / 0.3893 | 20.07 / 0.2414 | 20.07 / 0.2314 | 32.35 / 0.9241 |
| Fig. 5 · 03373 · 0.20 | 13.87 / 0.1397 | 17.22 / 0.3715 | 15.03 / 0.1900 | 15.65 / 0.2024 | 14.10 / 0.1441 | 14.89 / 0.1468 | 27.25 / 0.8440 |
| Fig. 5 · 01367 · 0.20 | 14.33 / 0.0996 | 18.74 / 0.3508 | 15.77 / 0.1561 | 16.59 / 0.1792 | 14.57 / 0.1053 | 15.30 / 0.1065 | 30.55 / 0.8705 |
| Fig. 6a · 022 mito · 0.05 | 26.02 / 0.6695 | 21.63 / 0.4603 | 29.02 / 0.7337 | 29.36 / 0.7788 | 27.58 / 0.6873 | 27.81 / 0.7049 | 30.35 / 0.8186 |
| Fig. 6a · 022 mito · 0.10 | 20.00 / 0.3398 | 18.42 / 0.1940 | 24.83 / 0.6635 | 24.24 / 0.6702 | 21.40 / 0.3949 | 23.88 / 0.5882 | 27.96 / 0.7310 |
| Fig. 6a · 022 mito · 0.20 | 13.98 / 0.1103 | 13.91 / 0.0542 | 17.95 / 0.3562 | 19.31 / 0.4778 | 15.06 / 0.1308 | 20.10 / 0.4310 | 26.60 / 0.6930 |
| Fig. 6b · 022 actin · measured | 30.54 / 0.5836 | 28.30 / 0.5235 | 35.20 / 0.8066 | 36.07 / 0.8454 | 33.59 / 0.8048 | 35.30 / 0.8121 | 36.65 / 0.8560 |
| Fig. 6c · 022 mito · measured | 21.20 / 0.3485 | 19.48 / 0.2216 | 24.91 / 0.6253 | 24.11 / 0.6317 | 22.50 / 0.4824 | 23.05 / 0.4828 | 27.53 / 0.6836 |

</details>

**Measured STM (Figure 4, qualitative only).** The three acquired scans have no paired clean reference, so PSNR and SSIM are undefined for this comparison. On Z00170, DONER weakens horizontal marks around dark spots, with some loss of local contrast. On Z00344 it weakens the streak through the central bright lobe. On Z00670 it attenuates lines across a bright band and the darker area below it while retaining the boundary.

## Appendix results

### Model size and restoration accuracy (Figure S1)

Counts include fixed weights used in the forward pass. Each domain-specific DONER denoiser has **5,623,796 parameters**, 83.7% fewer than CLIPDenoising’s 34,515,043. CLIPDenoising is the strongest baseline by mean PSNR on the three evaluations in the [summary table](#main-paper-results). The table below counts parameters per model, not the sum of the three DONER checkpoints.

| Restoration model | Parameters during denoising |
| --- | --- |
| DEAL | 572,323 |
| RNINet | 11,565,953 |
| CLIPDenoising | 34,515,043 |
| IDF | 49,021 |
| CENSURE | 17,008,256 |
| DONER | 5,623,796 |

SDXL uses 3,468,837,867 additional parameters offline to generate training textures. Counting it once alongside one DONER denoiser gives 3,474,461,663 parameters. Restoring an image uses only the denoiser. [Exact counts](results/parameter_counts.csv).

### Per-image results (Figures S2–S11)

The appendix contains **32 selected examples**: nine simulated STM, nine infrared, nine synthetic confocal and five measured confocal. They were selected because DONER scored highest among the six denoisers on **both** per-image metrics. Use the main tables for validation-set means. Each row below reproduces the appendix’s figure and panel, input score and all six model scores as **PSNR (dB) / SSIM**. The reference itself is not scored.

<details>
<summary>Figures S2–S4: Simulated STM (9 examples)</summary>

| Panel · reference · condition | Input | DEAL | RNINet | CLIPDenoising | IDF | CENSURE | DONER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2a · 00066 · grain + short streaks · 0.05 | 26.01 / 0.7790 | 22.12 / 0.7293 | 34.30 / 0.9603 | 35.06 / 0.9715 | 26.65 / 0.9124 | 26.96 / 0.9181 | 44.95 / 0.9883 |
| S2b · 00179 · grain + short streaks · 0.05 | 25.79 / 0.8271 | 22.32 / 0.7700 | 33.51 / 0.9599 | 34.54 / 0.9695 | 26.25 / 0.9079 | 26.37 / 0.9095 | 45.84 / 0.9886 |
| S2c · 00086 · thin scan-line stripes · 0.05 | 26.17 / 0.5221 | 22.41 / 0.4898 | 37.48 / 0.9893 | 35.98 / 0.9790 | 26.53 / 0.5622 | 26.24 / 0.5205 | 39.97 / 0.9945 |
| S3a · 00131 · sparse short streaks · 0.10 | 20.81 / 0.9326 | 19.25 / 0.8713 | 25.74 / 0.9464 | 31.30 / 0.9576 | 20.88 / 0.9510 | 23.32 / 0.9449 | 42.06 / 0.9875 |
| S3b · 00086 · grain + short streaks · 0.10 | 22.46 / 0.5934 | 20.40 / 0.5836 | 28.15 / 0.9536 | 32.28 / 0.9603 | 23.18 / 0.8461 | 24.67 / 0.9393 | 43.36 / 0.9824 |
| S3c · 00199 · grain + short streaks · 0.10 | 20.26 / 0.5443 | 18.73 / 0.5146 | 25.31 / 0.9148 | 29.57 / 0.9321 | 20.80 / 0.7680 | 22.67 / 0.9013 | 37.87 / 0.9744 |
| S4a · 00086 · fine grain · 0.20 | 13.96 / 0.0454 | 14.47 / 0.0508 | 28.02 / 0.6339 | 34.02 / 0.9270 | 14.90 / 0.0567 | 21.98 / 0.2590 | 37.24 / 0.9908 |
| S4b · 00199 · correlated grain · 0.20 | 13.85 / 0.0674 | 14.73 / 0.0819 | 23.56 / 0.5814 | 28.29 / 0.8103 | 14.50 / 0.0817 | 16.34 / 0.1139 | 33.09 / 0.9724 |
| S4c · 00161 · sparse short streaks · 0.20 | 12.80 / 0.8341 | 12.46 / 0.7019 | 12.95 / 0.7564 | 22.73 / 0.8644 | 12.85 / 0.9150 | 17.56 / 0.8393 | 35.86 / 0.9541 |

</details>

<details>
<summary>Figures S5–S7: Infrared (9 examples)</summary>

| Panel · reference · condition | Input | DEAL | RNINet | CLIPDenoising | IDF | CENSURE | DONER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S5a · 03765 · grain + column offsets · 0.05 | 26.12 / 0.5504 | 21.54 / 0.7436 | 27.97 / 0.6676 | 28.63 / 0.7496 | 26.55 / 0.5847 | 26.22 / 0.5503 | 37.99 / 0.9724 |
| S5b · 03373 · correlated grain · 0.05 | 26.00 / 0.5147 | 17.71 / 0.3178 | 32.26 / 0.8472 | 32.07 / 0.8376 | 27.35 / 0.6106 | 27.79 / 0.6055 | 33.90 / 0.9039 |
| S5c · 02344 · variable column offsets · 0.05 | 26.60 / 0.5509 | 23.22 / 0.7747 | 28.24 / 0.6646 | 29.13 / 0.7669 | 27.26 / 0.6154 | 26.74 / 0.5439 | 39.00 / 0.9728 |
| S6a · 03765 · grain + column offsets · 0.10 | 20.10 / 0.3006 | 20.66 / 0.5916 | 21.47 / 0.3833 | 21.68 / 0.4145 | 20.36 / 0.3122 | 20.35 / 0.3041 | 33.81 / 0.9483 |
| S6b · 03928 · fine grain · 0.10 | 19.99 / 0.1850 | 16.35 / 0.1405 | 30.71 / 0.6991 | 34.03 / 0.8816 | 21.14 / 0.2178 | 29.81 / 0.6228 | 35.85 / 0.9434 |
| S6c · 01706 · intermittent column offsets · 0.10 | 19.77 / 0.8820 | 21.37 / 0.7730 | 20.88 / 0.8019 | 21.87 / 0.7917 | 19.79 / 0.8640 | 21.05 / 0.8708 | 33.92 / 0.9272 |
| S7a · 03765 · grain + column offsets · 0.20 | 14.08 / 0.1292 | 18.59 / 0.3877 | 15.44 / 0.1796 | 15.81 / 0.1899 | 14.29 / 0.1323 | 14.71 / 0.1320 | 29.51 / 0.9016 |
| S7b · 03928 · fine grain · 0.20 | 13.97 / 0.0710 | 10.60 / 0.0519 | 21.51 / 0.2516 | 26.40 / 0.5455 | 14.94 / 0.0824 | 24.40 / 0.3762 | 31.92 / 0.9047 |
| S7c · 03373 · correlated grain · 0.20 | 13.96 / 0.1014 | 7.94 / 0.0485 | 17.44 / 0.1987 | 19.57 / 0.2796 | 14.63 / 0.1141 | 16.90 / 0.1645 | 27.45 / 0.7574 |

</details>

<details>
<summary>Figures S8–S10: Confocal, synthetic mixed grain (9 examples)</summary>

| Panel · reference · condition | Input | DEAL | RNINet | CLIPDenoising | IDF | CENSURE | DONER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S8a · 002 actin · 0.05 | 26.02 / 0.7526 | 14.92 / 0.5033 | 27.77 / 0.7051 | 29.25 / 0.8041 | 27.04 / 0.7846 | 27.45 / 0.7055 | 29.51 / 0.8315 |
| S8b · 038 mito · 0.05 | 26.02 / 0.9524 | 6.95 / 0.1887 | 25.58 / 0.9300 | 26.95 / 0.9535 | 26.11 / 0.9502 | 22.56 / 0.8347 | 27.11 / 0.9591 |
| S8c · 024 mito · 0.05 | 26.02 / 0.7776 | 18.63 / 0.4508 | 27.49 / 0.7287 | 28.76 / 0.8179 | 27.11 / 0.7752 | 26.78 / 0.6697 | 29.24 / 0.8539 |
| S9a · 069 actin · 0.10 | 20.00 / 0.4048 | 14.42 / 0.1778 | 24.97 / 0.6362 | 25.97 / 0.6952 | 20.98 / 0.4465 | 25.94 / 0.6325 | 28.06 / 0.7391 |
| S9b · 060 mito · 0.10 | 20.00 / 0.7710 | 9.88 / 0.2562 | 22.16 / 0.7899 | 22.17 / 0.8315 | 20.67 / 0.7899 | 21.20 / 0.6529 | 23.38 / 0.8378 |
| S9c · 003 actin · 0.10 | 20.00 / 0.4165 | 17.02 / 0.2303 | 25.59 / 0.6661 | 25.24 / 0.6787 | 21.20 / 0.4716 | 25.70 / 0.6380 | 28.11 / 0.7455 |
| S10a · 057 actin · 0.20 | 13.98 / 0.1200 | 12.33 / 0.0520 | 16.10 / 0.4848 | 18.08 / 0.6183 | 14.94 / 0.1431 | 19.35 / 0.5298 | 25.24 / 0.7870 |
| S10b · 047 mito · 0.20 | 13.98 / 0.0958 | 14.53 / 0.0638 | 18.58 / 0.2689 | 20.98 / 0.4162 | 15.08 / 0.1157 | 21.29 / 0.3590 | 26.70 / 0.5654 |
| S10c · 052 mito · 0.20 | 13.98 / 0.2936 | 11.66 / 0.1050 | 17.31 / 0.4382 | 17.80 / 0.4626 | 14.93 / 0.3234 | 19.79 / 0.4208 | 22.59 / 0.5721 |

</details>

<details>
<summary>Figure S11: Measured confocal pairs (5 examples)</summary>

| Panel · reference · condition | Input | DEAL | RNINet | CLIPDenoising | IDF | CENSURE | DONER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S11a · 002 actin | 19.68 / 0.3578 | 12.01 / 0.1591 | 22.49 / 0.5274 | 23.65 / 0.5708 | 20.47 / 0.3977 | 23.62 / 0.5249 | 25.85 / 0.6396 |
| S11b · 024 mito | 19.01 / 0.3529 | 16.54 / 0.1343 | 22.56 / 0.5762 | 21.71 / 0.5444 | 20.00 / 0.4130 | 21.12 / 0.4795 | 24.63 / 0.6323 |
| S11c · 069 actin | 22.21 / 0.4697 | 17.07 / 0.2426 | 27.01 / 0.6957 | 26.74 / 0.7194 | 23.09 / 0.5160 | 26.86 / 0.7020 | 27.46 / 0.7331 |
| S11d · 047 mito | 20.70 / 0.2430 | 20.21 / 0.1722 | 25.24 / 0.5254 | 24.13 / 0.4822 | 21.95 / 0.3240 | 22.93 / 0.3662 | 27.43 / 0.5962 |
| S11e · 052 mito | 16.09 / 0.2741 | 12.80 / 0.0544 | 18.98 / 0.4022 | 18.75 / 0.3868 | 16.90 / 0.2991 | 19.42 / 0.3953 | 21.98 / 0.5109 |

</details>

<details>
<summary>Evaluation data and sampling settings (Appendix A–C; Tables S1–S2)</summary>

The validation split holds out complete STM structures, infrared scene groups and confocal fields. The 50 STM maps cover two held-out structures × five bias voltages × five frames. M³FD contributes one infrared frame per validation scene group. Hagen Dataset 4 uses 79 fields: 4 for calibration, 25 for training, 25 for validation and 25 in reserve. Actin and mitochondria stay together in each split.

| Domain | Training size (H × W) | Example operator (Table S1) | Training strengths (Table S2) |
| --- | --- | --- | --- |
| STM | 300 × 300 | Sparse horizontal streaks; guide segments 16–48 pixels long and one pixel wide. | Log-uniform within [0.002, 0.035], [0.035, 0.12], [0.12, 0.25]; frequencies 30%, 35%, 35%. |
| Infrared | 768 × 1024 | Variable vertical column offsets; guide band widths of 1–3 pixels. | Multipliers 0.05, 0.10, 0.20 in shuffled three-update cycles. |
| Confocal | 128 × 128 | Short-correlated grain with weak horizontal dependence. | Target RMS uniform within [0.003, 0.05], [0.05, 0.12], [0.12, 0.25]; frequencies 45%, 30%, 25%. |

Evaluation uses the three nominal strengths 0.05, 0.10 and 0.20. For STM and infrared, the realized RMS can vary by pattern; synthetic mixed-grain confocal noise is rescaled to the target RMS. Measured confocal observations retain their acquisition noise.

</details>

## Full validation replay

The three released DONER checkpoints were evaluated on all **3,050 scored observations** with no training: 1,200 STM, 1,200 infrared, 600 synthetic confocal (four families) and 50 measured confocal pairs. Fifty additional confocal clean-input controls were evaluated separately and do not enter these means.

All **24/24 DONER entries** in Tables 1–4 match the paper at two decimal places for PSNR and four for SSIM. All 2,400 STM and infrared predictions match the archived predictions byte for byte. Across 650 confocal observations, the largest absolute per-image differences between the CPU replay and the archived CUDA scores were approximately 0.00001343 dB in PSNR and 0.0000005983 in SSIM. All **60/60** Table 5 margins also match at the reported precision; the five baseline scores come from the archived runs.

[Coverage by domain](validation/coverage.csv) · [Main-table comparison](validation/main_table_comparison.csv) · [Per-family margin check](validation/noise_family_gap_comparison.csv) · [Per-image differences](validation/case_differences.csv) · [Execution record](validation/verification.json)

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

## Reference data

Each domain has 50 training and 50 fixed validation references. Paths, source identifiers, split order, image dimensions, and file and float32 pixel hashes are in `data/index.csv`.

| Domain | Reference image | Representation |
|---|---|---|
| STM | Simulated MoS₂ height maps, normalized per map | Float32 NPY |
| Infrared | M3FD high-quality infrared frames used to synthesize noisy inputs | Original lossless 8-bit PNG |
| Confocal | Hagen actin and mitochondrial higher-SNR acquisitions, with 25 fields of view per split | Float32 NPY crops, already normalized by 65,535 |

The infrared and confocal references are captured images and may retain acquisition noise. Hagen dataset accession [10.5524/100888](https://doi.org/10.5524/100888).

Only reference images are distributed in this release. All paper scores and per-image records are linked in [Main-paper results](#main-paper-results).

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
