# Assignment 4: Object Detection Evolution — R-CNN to YOLO

**Name:** Kush  
**Student ID:** 202511052  
**Course:** IT549 — Deep Learning  

---

## Overview

This assignment explores the evolution of object detection architectures, from early region-proposal methods to modern single-shot detectors. Using the **Fruit Images for Object Detection** dataset (3 classes: Apple, Banana, Orange), we implement and compare key milestones in object detection history.

## Tasks

| # | Task | Description |
|---|------|-------------|
| Prep | Ground Truth Visualization | Load a random image, parse VOC XML annotations, and draw bounding boxes with class labels |
| 1 | IoU from Scratch | Implement `compute_iou()` and demonstrate on 3 box pairs (high, partial, zero overlap) |
| 2 | Selective Search | Use OpenCV's Selective Search to extract and visualize 200 region proposals |
| 3 | R-CNN Bottleneck | Crop 100 proposals individually through ResNet18; measure execution time |
| 4 | Fast R-CNN (RoI Pooling) | Single forward pass + RoI Pooling for 100 proposals; demonstrate **18.2x speedup** |
| 5 | Faster R-CNN | Pretrained Faster R-CNN inference with confidence filtering and visualization |
| 6 | NMS Implementation | Custom Non-Maximum Suppression using IoU function; before/after visualization |
| 7 | YOLOv8 Fine-Tuning | Fine-tune YOLOv8n on fruit dataset for 10 epochs; report mAP metrics |

## Key Results

### Timing Comparison (Task 3 vs Task 4)
| Approach | Time (100 proposals) | Speedup |
|----------|---------------------|---------|
| R-CNN (individual crops) | 1.772s | — |
| Fast R-CNN (RoI Pooling) | 0.097s | **18.2x** |

### Model Comparison (Task 7)
| Model | Inference (ms/img) | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-------------------|-----------|--------|--------|-----------|
| Faster R-CNN (pretrained) | 254.8 | N/A (COCO) | N/A (COCO) | N/A (COCO) | N/A (COCO) |
| YOLOv8n (pretrained) | 84.6 | 0.2633 | 0.1190 | 0.1527 | 0.0994 |
| **YOLOv8n (fine-tuned)** | **26.2** | **0.8434** | **0.7501** | **0.8875** | **0.6380** |

## Dataset

- **Source:** [Fruit Images for Object Detection](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection/data)
- **Classes:** Apple, Banana, Orange
- **Splits:** 240 train images, 60 test images
- **Annotations:** Pascal VOC XML format (xmin, ymin, xmax, ymax)

## How to Run

1. Download the dataset from Kaggle and place it in `fruit_dataset/`
2. Open `Assignment4_ObjectDetection.ipynb` in Jupyter Notebook
3. Run all cells sequentially (Kernel → Restart & Run All)

### Requirements
```
torch
torchvision
opencv-contrib-python
ultralytics
matplotlib
numpy
Pillow
pandas
pyyaml
```

## Repository Structure

```
├── .gitignore
├── README.md
├── Assignment4_ObjectDetection.ipynb
├── build_notebook_part1.py
├── yolo26n.pt
├── yolov8n.pt
├── fruit_dataset/
│   ├── train/
│   │   ├── apple_1.jpg
│   │   ├── apple_1.xml
│   │   ├── ...                          # 240 images + 240 XML annotations
│   │   ├── banana_1.jpg
│   │   ├── banana_1.xml
│   │   ├── ...
│   │   ├── mixed_1.jpg
│   │   ├── mixed_1.xml
│   │   ├── ...
│   │   ├── orange_1.jpg
│   │   └── orange_1.xml
│   └── test/
│       ├── apple_77.jpg
│       ├── apple_77.xml
│       ├── ...                          # 60 images + 60 XML annotations
│       ├── banana_77.jpg
│       ├── banana_77.xml
│       ├── ...
│       ├── mixed_21.jpg
│       ├── mixed_21.xml
│       ├── ...
│       ├── orange_77.jpg
│       └── orange_77.xml
├── fruit_yolo/
│   ├── dataset.yaml
│   ├── images/
│   │   ├── train/                       # 192 training images (.jpg)
│   │   ├── val/                         # 48 validation images (.jpg)
│   │   └── test/                        # 60 test images (.jpg)
│   └── labels/
│       ├── train/                       # 192 YOLO-format labels (.txt)
│       ├── val/                         # 48 YOLO-format labels (.txt)
│       └── test/                        # 60 YOLO-format labels (.txt)
├── yolo_runs/
│   └── fruit_yolov8n/
│       ├── args.yaml
│       ├── results.csv
│       ├── results.png
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│       ├── BoxF1_curve.png
│       ├── BoxPR_curve.png
│       ├── BoxP_curve.png
│       ├── BoxR_curve.png
│       ├── labels.jpg
│       ├── train_batch0.jpg
│       ├── train_batch1.jpg
│       ├── train_batch2.jpg
│       ├── val_batch0_labels.jpg
│       ├── val_batch0_pred.jpg
│       ├── val_batch1_labels.jpg
│       ├── val_batch1_pred.jpg
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── runs/
    └── detect/
        ├── val/
        │   ├── val_batch0_labels.jpg
        │   ├── val_batch0_pred.jpg
        │   ├── val_batch1_labels.jpg
        │   ├── val_batch2_labels.jpg
        │   └── val_batch2_pred.jpg
        ├── val2/
        └── val3/
```
