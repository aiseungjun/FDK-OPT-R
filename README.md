# FDK-OPT+R: Teacher-Free Real-Time Low-Dose Fluoroscopy Denoising via Risk-Calibrated Pseudo Supervision and Kalman-Form Fusion

Public release code for the paper experiments.

## Environment (tested)
- Python: `3.11.12`
- PyTorch: `2.7.0+cu128`
- Torchvision: `0.22.0+cu128`
- NumPy: `2.2.5`
- OpenCV (`opencv-python`): `4.11.0`
- h5py: `3.13.0`
- Pillow: `11.0.0`
- SciPy: `1.15.2`

## Repository layout
```text
paper_code/
  opt_flow_generater.py
  peak_sigma_estimate.py
  ablation_temp_power.py
  ablation_opt.py
  README.md
  EndovascularSurgery/
    data.py
    data_ood.py
    train.py
    calc_metric.py
    run_paper_experiments.py
    run_ood_experiments.py
    run_edge_experiments.py
    model/
    weiss/                 # data placeholder
    opt_label_weiss/       # generated OPT-flow pseudo labels + risk maps (for OPT+R training)
  Orthopedic/
    data.py
    train.py
    calc_metric.py
    run_paper_experiments.py
    model/
    jhu/                   # data placeholder
    opt_label_jhu/         # generated OPT-flow pseudo labels + risk maps (for OPT+R training)
  InterventionalCardiology/
    data.py
    data_ood.py
    train.py
    calc_metric.py
    run_paper_experiments.py
    run_ood_experiments.py
    run_edge_experiments.py
    model/
    cardio_data/           # data placeholder
    opt_label_cardiac/     # generated OPT-flow pseudo labels + risk maps (for OPT+R training)
```

## Dataset setup
Raw data is not included. Follow placeholder guides:
- `EndovascularSurgery/weiss/README.md`
- `Orthopedic/jhu/README.md`
- `InterventionalCardiology/cardio_data/README.md`

### Dataset sources
- `weiss`: Gherardini, M., Mazomenos, E., Menciassi, A., Stoyanov, D. Catheter segmentation in X-ray fluoroscopy using synthetic data and transfer learning with light U-nets. Computer Methods and Programs in Biomedicine 197, 105420 (2020). https://doi.org/10.1016/j.cmpb.2020.105420
- `jhu`: Grupp, R.B., Unberath, M., Gao, C., Hegeman, R.A., Murphy, R.J., Alexander, C.P., Otake, Y., McArthur, B.A., Armand, M., Taylor, R.H. Data associated with the publication: X-ray computed tomography and fluoroscopy collected during cadaveric periacetabular osteotomy experiments. Johns Hopkins Research Data Repository (2022), version 1.1. https://doi.org/10.7281/T1/C304HZ
- `cardio`: Kruzhilov, I., Mazanov, G., Ponomarchuk, A., Zubkova, G., Shadrin, A., Utegenov, R., Blinov, P., Bessonov, I. Coronarydominance: A dataset for coronary artery dominance and diagnosis from invasive coronary angiography. Scientific Data 12(341) (2025). https://doi.org/10.1038/s41597-025-04676-8

## How to run
### 0) Estimate and set `peak` / `sigma_read`
```bash
python peak_sigma_estimate.py
```
Copy each dataset result into:
- `EndovascularSurgery/data.py`: `DEFAULT_LOW_DOSE_CFG["peak"]`, `DEFAULT_LOW_DOSE_CFG["sigma_read"]`
- `Orthopedic/data.py`: `DEFAULT_LOW_DOSE_CFG["peak"]`, `DEFAULT_LOW_DOSE_CFG["sigma_read"]`
- `InterventionalCardiology/data.py`: `DEFAULT_LOW_DOSE_CFG["peak"]`, `DEFAULT_LOW_DOSE_CFG["sigma_read"]`

### 1) Generate OPT-flow pseudo labels + risk maps (for OPT+R training)
```bash
python opt_flow_generater.py --datasets weiss jhu cardio
```

### 2) Run main paper experiments
```bash
python EndovascularSurgery/run_paper_experiments.py --root EndovascularSurgery
python Orthopedic/run_paper_experiments.py --root Orthopedic
python InterventionalCardiology/run_paper_experiments.py --root InterventionalCardiology --dataset cardiac
```

### 3) Evaluate OOD and edge robustness
```bash
python EndovascularSurgery/run_ood_experiments.py --root EndovascularSurgery
python EndovascularSurgery/run_edge_experiments.py --root EndovascularSurgery

python InterventionalCardiology/run_ood_experiments.py --root InterventionalCardiology --dataset_kind cardiac
python InterventionalCardiology/run_edge_experiments.py --root InterventionalCardiology --dataset_kind cardiac
```

Notes:
- OOD scripts default to a fixed test subset (`500` frames). To use full test set, set `--test_max_samples 0`.
- Main paper evaluation (`run_paper_experiments.py` + `calc_metric.py`) for cardio uses full test set by default.

### 4) Ablations
```bash
python ablation_temp_power.py --mode opt+r
python ablation_opt.py --datasets weiss jhu cardio
```

## Per-folder file roles
- `data.py`: dataset split/loader and low-dose simulation.
- `data_ood.py`: OOD corruption dataset wrapper used by OOD evaluation.
- `train.py`: training entrypoint.
- `calc_metric.py`: PSNR/SSIM/latency evaluation and summary CSV generation.
- `run_paper_experiments.py`: main paper run script.
- `run_ood_experiments.py`: OOD robustness evaluation.
- `run_edge_experiments.py`: edge-preservation evaluation.
- `model/`: model definitions (`UNet`, `FDK`, baselines).
