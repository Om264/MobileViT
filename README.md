# MobileViT for Plant Disease Classification

A complete, dataset-agnostic PyTorch pipeline for classifying plant diseases from leaf images with
**MobileViT** (via [timm](https://github.com/huggingface/pytorch-image-models)). It covers training,
evaluation, prediction and ONNX export, and runs on CPU, one GPU, or many GPUs.

## Features

- **Any image-folder dataset**: PlantVillage, PlantDoc, PlantWild or your own field images. The number of classes, class names, input size and normalisation are detected automatically.
- **Any MobileViT variant** (or any other timm model) with ImageNet-pretrained weights: `--model mobilevit_s`.
- **Leakage-safe splitting**: stratified by default, or grouped by leaf/plant ID with `--group_regex`.
- **Scalable training**: multi-GPU (DDP via `torchrun`), mixed precision (fp16/bf16), gradient accumulation, EMA, resume, early stopping.
- **Imbalance and robustness tools**: class-weighted loss, label smoothing, Mixup/CutMix, optional stronger augmentation.
- **Research-ready outputs**: per-class precision/recall/F1, macro-F1, confusion matrices, training curves, saved splits and arguments.
- **Deployment**: self-contained checkpoints (class names + preprocessing inside) and ONNX export.

## Project structure

```
mobilevit-plant-disease/
├── README.md
├── requirements.txt
├── train.py                  # entry point: train
├── evaluate.py               # entry point: evaluate a checkpoint on a folder
├── predict.py                # entry point: predict on an image / folder
├── export_onnx.py            # entry point: export to ONNX
├── plant_cls/
│   ├── config.py             # command-line arguments
│   ├── data.py               # dataset discovery, splits, transforms
│   ├── model.py              # model factory, EMA, checkpoint loading
│   ├── engine.py             # training loop (AMP, DDP, accumulation, resume)
│   ├── metrics.py            # evaluation, reports, plots
│   ├── inference.py          # evaluate / predict / export logic
│   └── utils.py              # seeding, device, distributed helpers
├── scripts/
│   ├── train_single_gpu.sh
│   ├── train_multi_gpu.sh
│   └── make_dummy_dataset.py # tiny synthetic dataset for smoke tests
└── tests/
    └── test_splits.py
```

## Installation

Python 3.9+ is required.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version first
(see https://pytorch.org/get-started/locally/), then `pip install -r requirements.txt`.

## Dataset format

Point `--data_dir` at either layout (detected automatically):

**A. Already split**
```
data/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg        # "valid" or "validation" also work
└── test/<class_name>/*.jpg       # optional
```

**B. Single folder**: split automatically with `--val_split` (default 0.15) and `--test_split` (default 0.15)
```
data/
├── Tomato___Early_blight/*.jpg
├── Tomato___healthy/*.jpg
└── ...
```

### Avoiding data leakage (important)

Datasets such as PlantVillage contain several photos of the same leaf. A random split can put
near-duplicates in both train and test and inflate accuracy. If your filenames encode the leaf or
plant ID, use `--group_regex` with **one capture group** that extracts it from the file name
(without extension). Images with the same ID are then always kept in the same split:

```bash
# files like  leaf0042_3.jpg  ->  group "leaf0042"
python train.py --data_dir data/PlantVillage --group_regex '^(.+)_\d+$'
```

## Quick start

```bash
# 1. Train (downloads ImageNet-pretrained MobileViT-S weights on first run)
python train.py --data_dir /path/to/data --model mobilevit_s --epochs 30

# 2. Evaluate on a held-out folder
python evaluate.py --checkpoint runs/mobilevit_s/best.pt --data_dir /path/to/test

# 3. Predict
python predict.py --checkpoint runs/mobilevit_s/best.pt --input leaf.jpg --topk 3
python predict.py --checkpoint runs/mobilevit_s/best.pt --input folder/ --output_csv preds.csv

# 4. Export for deployment
python export_onnx.py --checkpoint runs/mobilevit_s/best.pt
```

### Multi-GPU

```bash
torchrun --nproc_per_node=4 train.py --data_dir /path/to/data --scale_lr
# or: NGPUS=4 bash scripts/train_multi_gpu.sh /path/to/data
```

`--batch_size` is **per GPU**. Use `--scale_lr` to scale the learning rate with the global batch size,
and `--accum_steps N` to simulate larger batches on small GPUs.

### Smoke test without real data

```bash
python scripts/make_dummy_dataset.py --out data/dummy --classes 4 --per_class 60
python train.py --data_dir data/dummy --model mobilevit_xxs --no-pretrained --epochs 2 --batch_size 8 --workers 0
pytest -q          # optional: unit tests for the splitting logic (needs pytest)
```

## Choosing a model

| `--model`                          | Params (approx.) | Notes                              |
|------------------------------------|------------------|------------------------------------|
| `mobilevit_xxs`                    | 1.3 M            | smallest, fastest, edge devices    |
| `mobilevit_xs`                     | 2.3 M            | good accuracy/size trade-off       |
| `mobilevit_s`                      | 5.6 M            | default                            |
| `mobilevitv2_050` to `_200`        | 1.4 M to 18 M    | MobileViT-v2 family, scalable width |

Any other timm model name also works (for example `efficientnet_b0` or `convnext_tiny`), which makes
baseline comparisons easy. List available names with `python -c "import timm; print(timm.list_models('mobilevit*'))"`.

## Useful options

| Option | Purpose |
|--------|---------|
| `--img_size 224` | override the native input size (256) |
| `--class_weights` | inverse-frequency loss weights for imbalanced classes |
| `--aug strong` | adds TrivialAugment on top of the default augmentation |
| `--mixup 0.2 --cutmix 1.0` | Mixup / CutMix (off by default; CutMix can hide small lesions) |
| `--ema_decay 0.999` | evaluate and save an EMA of the weights |
| `--amp_dtype bf16` | bfloat16 instead of fp16 (Ampere GPUs and newer) |
| `--channels_last` | memory-format optimisation on GPUs |
| `--opt sgd --lr 0.01` | any timm optimizer |
| `--patience 10` | early stopping on validation macro-F1 (0 disables) |
| `--resume runs/<exp>/last.pt` | continue an interrupted run |
| `--no-pretrained` | train from scratch (or if weights cannot be downloaded) |

Run `python train.py --help` for the full list.

## Outputs

Every run writes to `runs/<exp_name>/` (default `exp_name` is the model name):

| File | Content |
|------|---------|
| `best.pt` | best weights by validation macro-F1, plus class names and preprocessing (used by evaluate/predict/export) |
| `last.pt` | full training state for `--resume` |
| `val_report.txt/.json`, `test_report.txt/.json` | per-class precision, recall, F1, support |
| `*_confusion_matrix.csv/.png` | confusion matrices |
| `training_curves.png`, `history.json` | loss, accuracy, macro-F1 per epoch |
| `results.json` | final metrics summary |
| `splits.json`, `class_names.json`, `args.json` | everything needed to reproduce the run |

## Tips for plant-disease research

- **Report macro-F1 and per-class metrics**, not only accuracy, since classes are usually imbalanced.
- **Lab-to-field gap**: models trained on clean lab images (PlantVillage) often drop sharply on field photos. Evaluate on a separate field dataset such as PlantDoc, use `--aug strong`, and consider mixing in field images.
- **Compare against baselines** (for example `efficientnet_b0`, `resnet50`) with the same splits, using `--exp_name` to keep runs separate and a fixed `--seed`.
- **Reproducibility**: splits depend on `--seed`; `splits.json` stores the exact files used.

## Troubleshooting

- **CUDA out of memory**: lower `--batch_size` and raise `--accum_steps`, or choose a smaller model.
- **Pretrained weights cannot be downloaded** (offline machine): use `--no-pretrained`, or download the weights on another machine and copy the Hugging Face cache (`~/.cache/huggingface`) over.
- **`Validation set is empty`**: provide a `val/` folder or increase `--val_split`.
- **Odd batch size error with Mixup/CutMix**: use an even `--batch_size`.
- **Slow data loading**: raise `--workers`, and keep the dataset on fast local storage.

## References

- S. Mehta and M. Rastegari, *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*, ICLR 2022.
- S. Mehta and M. Rastegari, *Separable Self-attention for Mobile Vision Transformers* (MobileViT-v2), 2022.
