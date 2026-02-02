# Efficient-HG-YOLOv9: Task-Oriented Object Detection for EV Battery Disassembly

This repository contains the implementation of **Efficient-HG-YOLOv9**, a task-oriented object detection architecture designed for perception in autonomous electric vehicle (EV) battery disassembly.

The work demonstrates that aligning architectural design with domain constraints improves recall for small, safety-critical components more effectively than simply scaling generic object detectors.

---

## 🔬 Problem Motivation

Internal EV battery assemblies present severe challenges for object detection:

- Dense metallic clutter
- Specular reflections and glare
- Severe occlusions
- Small fasteners and elongated cables
- Safety-critical components tightly packed together

Conventional benchmarks (COCO, Open Images) do not represent these conditions.

---

## 🧩 Proposed Architecture

Efficient-HG-YOLOv9 is a composition of:

- EfficientNet backbone for compound multi-scale feature extraction
- Hourglass-inspired spatial refinement blocks (HGBlocks)
- Feature Attention (channel + spatial)
- Integrated into the YOLOv9 detection framework

The design specifically targets small-object recall in cluttered industrial scenes.

---

## 🗂 Dataset

This work uses the publicly available dataset:

**EV-Battery-Components**  
https://universe.roboflow.com/mour250027044/ev-battery-components-edfw3

- 1,014 images
- 4,694 annotated instances
- 8 safety-critical classes
- Polygon annotations from real EV teardown videos

---

## 📊 Key Results (from paper)

| Model | Recall | mAP50 | mAP50-95 |
|------|-------|------|---------|
| YOLOv9c (baseline) | 0.564 | 0.584 | 0.424 |
| **Efficient-HG-YOLOv9** | **0.637** | **0.613** | **0.448** |

Largest improvements occur for nuts, cables, connectors, and busbars — critical for robotic safety.

---

## ⚙️ Installation

git clone https://github.com/mour250027044/efficientnet-yolov9-object-dectetion-tool
cd efficientnet-yolov9-object-dectetion-tool
pip install -r requirements.txt
