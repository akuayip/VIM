# Head Detection Project — Vim + Detectron2

## Struktur Folder

```
HEAD_DETECTION_PROJECT/
├── install.sh                      ← jalankan ini pertama kali
├── README.md
├── libs/
│   ├── causal-conv1d/              ← install: cd libs/causal-conv1d && pip install -e .
│   ├── mamba-1p1p1/                ← install: cd libs/mamba-1p1p1 && pip install -e .
│   └── detectron2_vim/             ← install: cd libs/detectron2_vim && pip install -e .
└── project/
    ├── requirements.txt
    ├── checkpoints/
    │   └── vim_tiny_pretrained.pth ← TARUH PRETRAINED DI SINI
    ├── data/
    │   ├── train/
    │   │   ├── images/             ← TARUH GAMBAR TRAINING DI SINI
    │   │   └── annotations.json    ← TARUH ANOTASI COCO DI SINI
    │   └── val/
    │       ├── images/             ← TARUH GAMBAR VALIDASI DI SINI
    │       └── annotations.json
    ├── configs/
    │   ├── models/faster_rcnn_vimdet.py
    │   ├── data/head_coco_loader.py
    │   ├── ofat/                    ← config eksperimen OFAT
    │   └── head_detection_baseline.py  ← edit NUM_TRAIN_IMAGES
    ├── tools/
    │   ├── train_net.py
    │   ├── eval_efficiency.py
    │   └── eval_effectiveness.py
    ├── scripts/
    │   ├── analyze_ofat.py
    │   └── compare_models.py
    └── outputs/                    ← hasil training tersimpan di sini
```

## Cara Install
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

```bash
chmod +x install.sh
./install.sh
```

## Cara Training
```bash
cd project

# Edit jumlah data training
# nano configs/head_detection_baseline.py  → ubah NUM_TRAIN_IMAGES

# Single run
python tools/train_net.py \
    --config-file configs/head_detection_baseline.py \
    --num-gpus 1

```
