# Product Requirements Document (PRD)

## Frugal Edge-AI Autonomous Precision Spraying Rover

| Field | Detail |
|---|---|
| **Document Version** | 3.0 |
| **Date** | 7 March 2026 |
| **Classification** | Academic — B.Tech CSE Final-Semester Capstone Project |
| **Institution** | REVA University, Bengaluru, Karnataka, India |
| **Team** | Chirag · Sayed · Omkar · Vinayak |
| **Budget Ceiling** | INR 18,000 (hard cap) |
| **Development Window** | 12 weeks |
| **Target Publication** | IEEE International Conference on AgriTech / Precision Agriculture |

---

## Table of Contents

1. [Executive Summary & Product Vision](#1-executive-summary--product-vision)
2. [Problem Statement & Target Audience](#2-problem-statement--target-audience)
3. [Objectives & Success Metrics](#3-objectives--success-metrics)
4. [System Architecture](#4-system-architecture)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Bill of Materials & Estimated Budget Breakdown](#7-bill-of-materials--estimated-budget-breakdown)
8. [Development Timeline](#8-development-timeline-12-week-phased-approach)
9. [Academic Validation & Testing Plan](#9-academic-validation--testing-plan)
10. [Risk Register & Mitigation](#10-risk-register--mitigation)
11. [References](#11-references)
12. [Revision History](#12-revision-history)

---

## 1. Executive Summary & Product Vision

### 1.1 Executive Summary

This document specifies the product requirements for the **Frugal Edge-AI Autonomous Precision Spraying Rover** — a low-cost, software-defined precision agriculture prototype designed to detect tomato leaf diseases (Early Blight caused by *Alternaria solani* and Late Blight caused by *Phytophthora infestans*) in real time at the field edge and actuate targeted foliar spraying with zero cloud dependency.

The system embodies **Frugal Innovation** (Jugaad Engineering): rather than employing expensive motorized 3-DOF robotic arms, variable-aperture nozzles, or LiDAR-based SLAM, it achieves equivalent Variable Rate Application (VRA) functionality through *software-defined spray duration modulation* and *continuous kinematic nozzle positioning* via a 1-DOF Z-axis linear actuator (NEMA 17 stepper motor + GT2 belt drive on a 2040 V-slot rail).

A quantized ResNet-18 Convolutional Neural Network, pre-trained on ImageNet and fine-tuned on a rigorously cleaned 99,152-image crop disease dataset (achieving **95.34% test accuracy** across 18 disease classes), is deployed on a Raspberry Pi 4 Model B (4 GB). The inference pipeline processes 224×224 RGB frames captured under natural sunlight, converts the detected disease bounding-box Y-center pixel into a physical travel distance via a kinematic pixel-to-step formula, commands the NEMA 17 stepper gantry to position a single nozzle at the target height, computes severity-proportional spray duration, and actuates the 12V solenoid relay — all within a target latency budget of ≤750 ms per frame.

The rover chassis is a 4WD DC geared-motor platform governed by an Arduino Nano / ESP32 microcontroller via an L298N or BTS7960 H-bridge motor driver. HC-SR04 ultrasonic sensors provide rudimentary row-following and obstacle-avoidance capability.

The entire system is designed, fabricated, and field-validated within a **12-week timeline** and an **INR 18,000 budget**, making it reproducible by student teams and extensible for small-holder farming contexts across South and Southeast Asia.

### 1.2 Product Vision

> **"Deliver farm-gate intelligence — not cloud intelligence — so that a single tomato farmer with a one-acre plot can autonomously detect and treat Early/Late Blight with spatially targeted, severity-proportional spraying, at a total system cost below the price of two bags of premium fungicide."**

### 1.3 Core Philosophy — Frugal Innovation

| Conventional Approach | Frugal Replacement | Engineering Rationale |
|---|---|---|
| 3-DOF robotic arm with servo-driven nozzle | **1-DOF Z-axis linear actuator** (NEMA 17 + GT2 belt + 2040 V-slot, 80 cm travel) | Continuous kinematic pixel-to-step targeting with a single nozzle; far cheaper than multi-zone solenoid array |
| Variable-pressure pump / motorized aperture nozzle | Fixed-pressure pump + **software-defined spray duration** | Pixel-area of bounding box → relay-open duration (ms); no additional hardware |
| GPU compute module (Jetson Nano / Xavier) | Raspberry Pi 4 (4 GB) + INT8/FP16 quantized model | Quantization reduces model footprint; Pi 4 is 3× cheaper than Jetson Nano |
| LiDAR SLAM navigation | HC-SR04 ultrasonic array + PID-based row-following | Adequate for straight row-crop geometry at walking speed |
| Cloud-based inference pipeline | Fully offline edge inference | Zero connectivity dependency; sub-second latency |

---

## 2. Problem Statement & Target Audience

### 2.1 Problem Statement

Tomato (*Solanum lycopersicum*) is one of India's most widely cultivated horticultural crops, with an estimated 21.18 million metric tonnes produced annually (NHB, 2023). Early Blight (*Alternaria solani*) and Late Blight (*Phytophthora infestans*) together account for yield losses of **25–75%** when left untreated, with Late Blight capable of destroying an entire field within 7–10 days under favorable conditions (high humidity, 15–25°C).

Current pest management on small-to-medium farms in India suffers from three compounding failures:

1. **Late Detection:** Farmers identify blight visually only after >30% canopy damage, past the optimal treatment window.
2. **Uniform Broadcast Spraying:** Knapsack sprayers deliver a blanket dose irrespective of infection severity or spatial distribution, leading to:
   - 40–60% fungicide wastage (Gil et al., 2014).
   - Soil and groundwater contamination.
   - Accelerated pathogen resistance (FRAC, 2022).
3. **Cost Barrier to Precision Agriculture:** Commercial precision spraying platforms (e.g., DJI Agras T40, John Deere See & Spray) cost USD 10,000–50,000+, economically inaccessible to >86% of Indian farmers who operate on holdings <2 hectares (Agricultural Census, 2015-16).

### 2.2 Target Audience

| Persona | Description | Pain Point | How the Rover Addresses It |
|---|---|---|---|
| **Primary:** Small/Medium Tomato Farmer | 0.5–5 acre holding, annual input budget < INR 50,000 | Cannot afford scouts or precision tech; relies on calendar-based blanket spraying | Autonomous detection + targeted spray at < INR 18,000 build cost |
| **Secondary:** Agricultural Extension Officer | State/district agriculture department | Lacks scalable tools to demonstrate IPM best practices | Low-cost demonstrator unit for field days and training |
| **Tertiary:** AgriTech Researcher / Student | University agriculture & CS departments | Needs an open, reproducible edge-AI+actuation testbed | Fully documented, open-source hardware/software stack |

### 2.3 Use-Case Environment

| Parameter | Specification |
|---|---|
| Target Crop | Tomato (*Solanum lycopersicum*), determinate and semi-determinate varieties |
| Target Diseases | Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*) |
| Row Spacing | 60–90 cm (standard Indian row-crop geometry) |
| Plant Height | 45–120 cm (vegetative to fruiting stage) |
| Operating Conditions | Natural daylight (>2,000 lux), ambient temp 15–40°C, dry to moderate humidity |
| Terrain | Flat to gently sloping (<5°), red laterite / alluvial loam |
| Operating Speed | 0.1–0.3 m/s (walking-speed crawl) |
| Spray Agent | Mancozeb 75% WP or Metalaxyl-M + Mancozeb (premixed) |

---

## 3. Objectives & Success Metrics

### 3.1 Project Objectives

| ID | Objective | Category |
|---|---|---|
| O-1 | Train and quantize a CNN model achieving ≥90% field-validated accuracy for Early/Late Blight detection on tomato leaves under natural sunlight | Edge-AI Vision |
| O-2 | Demonstrate software-defined VRA via spray-duration modulation, eliminating variable-pressure hardware | Precision Spraying |
| O-3 | Achieve kinematic nozzle targeting accuracy ≤±15 mm (Y-pixel → stepper position) via pixel-to-step conversion | Kinematic Actuator Targeting |
| O-4 | Build and field-test the complete rover within INR 18,000 and 12 weeks | Frugal Engineering |
| O-5 | Produce a publishable field-validation dataset and results suitable for an IEEE-format conference paper | Academic Output |

### 3.2 Success Metrics (Key Performance Indicators)

| KPI | Target | Measurement Method | Rationale |
|---|---|---|---|
| **Edge Inference FPS** | ≥ 2 FPS (≤500 ms/frame) | Timestamp delta averaged over 500 frames on Pi 4 | Ensures real-time-enough detection at rover crawl speed of 0.2 m/s |
| **Model Accuracy (Lab/Test Set)** | ≥ 95% | sklearn `classification_report` on held-out 20,042-image test set | Baseline established: **95.34%** (ResNet-18, Epoch 30) |
| **Model Accuracy (Field-Validated)** | ≥ 85% | Manual ground-truth annotation of 200+ field-captured frames | Accounts for domain shift (lab images → real field) |
| **Kinematic Targeting Accuracy** | ≤ ±15 mm positional error | Water-sensitive paper test at 5 known plant heights (0, 200, 400, 600, 800 mm); measure offset with ruler | Nozzle position matches predicted Y-height from pixel-to-step formula |
| **Spray Duration Proportionality** | R² ≥ 0.80 (bbox area vs. spray duration) | Linear regression on logged bbox-area / relay-duration pairs | Validates VRA without variable-pressure hardware |
| **Obstacle Avoidance Success** | ≥ 90% | 50-trial obstacle course (static objects in row path) | Rover stops or reroutes without collision |
| **End-to-End Latency** | ≤ 750 ms | Capture → Inference → Relay Actuation, measured with logic analyzer | Total pipeline must complete before rover passes the plant |
| **System Uptime** | ≥ 45 min continuous operation | Timed field run on full battery charge | Sufficient for a 0.5-acre pass |
| **Total Build Cost** | ≤ INR 18,000 | Itemized BoM with purchase receipts | Hard budget constraint |
| **Fungicide Reduction** | ≥ 30% vs. uniform broadcast | Volumetric comparison (targeted run vs. blanket spray on same row) | Primary agronomic value proposition |

### 3.3 Accuracy Decomposition

The system tracks accuracy at three levels to isolate failure modes:

$$
\eta_{\text{system}} = \eta_{\text{detection}} \times \eta_{\text{kinematic-targeting}} \times \eta_{\text{actuation}}
$$

| Stage | Symbol | Target | Notes |
|---|---|---|---|
| Disease Detection (CNN) | $\eta_{\text{detection}}$ | ≥ 0.85 (field) | Precision & Recall on Blight classes |
| Kinematic Targeting (Y-pixel → stepper steps → mm) | $\eta_{\text{kinematic-targeting}}$ | ≤ ±15 mm error | Physics-based formula; validated with water-sensitive paper |
| Relay Actuation Fidelity | $\eta_{\text{actuation}}$ | ≥ 0.98 | Hardware relay response; near-deterministic |
| **Composite System Accuracy** | $\eta_{\text{system}}$ | **≥ 0.75** | Detection × targeting × actuation |

---

## 4. System Architecture

### 4.1 High-Level Architecture Diagram (Textual)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRUGAL EDGE-AI SPRAYING ROVER                    │
│                                                                         │
│  ┌──────────────┐    USB     ┌────────────────────────────────────┐     │
│  │  RGB Camera   │──────────▶│        RASPBERRY PI 4 (4GB)        │     │
│  │  (Logitech    │           │                                    │     │
│  │   C270 /      │           │  ┌──────────────────────────────┐  │     │
│  │   Pi Camera)  │           │  │  Quantized ResNet-18 (INT8)  │  │     │
│  └──────────────┘           │  │  Input: 224×224×3 RGB         │  │     │
│                              │  │  Output: 3-class softmax      │  │     │
│                              │  │  + y_center_px (0–223)        │  │     │
│                              │  └──────────┬───────────────────┘  │     │
│                              │             │                      │     │
│                              │  ┌──────────▼───────────────────┐  │     │
│                              │  │  SPRAY DECISION ENGINE        │  │     │
│                              │  │                               │  │     │
│                              │  │  KinematicMapper              │  │     │
│                              │  │  y_px → H_mm:                 │  │     │
│                              │  │    H = (y/224) × 300 mm       │  │     │
│                              │  │  H_mm → steps:                │  │     │
│                              │  │    steps = round(H_mm × 80)   │  │     │
│                              │  │                               │  │     │
│                              │  │  BBox Area → Duration(ms)     │  │     │
│                              │  │    d = α·A_pixel + β          │  │     │
│                              │  │    Clamp: [50ms, 500ms]       │  │     │
│                              │  └──────────┬───────────────────┘  │     │
│                              │             │ UART JSON / GPIO      │     │
│                              └─────────────┼──────────────────────┘     │
│                                            │                            │
│                    ┌───────────────────────┼───────────────────────┐    │
│                    │                       ▼                       │    │
│          ┌─────────────────┐    ┌─────────────────────┐           │    │
│          │  ARDUINO NANO   │    │  RELAY MODULE (2ch)  │           │    │
│          │                 │    │                     │           │    │
│          │  • Motor Control│    │  CH1 → Nozzle solenoid          │    │
│          │  • Ultrasonic   │    │  CH2 → Pump Motor    │           │    │
│          │    Sensors (×2) │    │                     │           │    │
│          │  • Row-Following│    └─────────┬───────────┘           │    │
│          │  • Obstacle     │              │ 12V Switched           │    │
│          │    Avoidance    │              ▼                       │    │
│          │  • Stepper GOTO │    ┌───────────────────┐             │    │
│          │    (JSON UART)  │    │  Z-AXIS GANTRY    │             │    │
│          └────────┬────────┘    │  2040 V-slot 80cm │             │    │
│                   │ STEP/DIR    │                   │             │    │
│                   ▼             │  [TOP]  ·         │             │    │
│          ┌────────────────┐     │         │         │             │    │
│          │ A4988/DRV8825  │     │  [MID]  ● ←── Single nozzle    │    │
│          │ Stepper Driver │     │  (movable)        │             │    │
│          └───────┬────────┘     │         │         │             │    │
│                  │              │  [BOT]  ·         │             │    │
│                  ▼              │                   │             │    │
│          ┌────────────────┐     │ NEMA 17 + GT2     │             │    │
│          │  NEMA 17        │     │ belt + pulley     │             │    │
│          │  Stepper Motor  │     └───────────────────┘             │    │
│          │  80 steps/mm    │                                       │    │
│          └────────────────┘      12V Supply ← Battery (LiPo/     │    │
│                                              Lead-Acid)           │    │
│          ┌────────────────┐                                       │    │
│          │  L298N / BTS   │  ┌──────────────┐  ┌──────────────┐  │    │
│          │  Motor Driver  │  │ HC-SR04 (L)  │  │ HC-SR04 (R)  │  │    │
│          │  M1 M2 M3 M4   │  └──────────────┘  └──────────────┘  │    │
│          │  4WD DC Motors │   Ultrasonic Rangefinders             │    │
│          └────────────────┘                                       │    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow Pipeline

```
Frame Capture (USB Camera @ 640×480)
        │
        ▼
Pre-processing (Resize 256→CenterCrop 224, Normalize ImageNet μ/σ)
        │
        ▼
CNN Inference (Quantized ResNet-18, ~200–400ms on Pi 4 CPU/NNAPI)
        │
        ├──► Class Prediction: {healthy, early_blight, late_blight, ...}
        │
        ├──► Confidence Score: P(class) ∈ [0, 1]
        │        │
        │        └──► Threshold Gate: P ≥ 0.70 → proceed; else → skip
        │
        ├──► Bounding Box: (x_center, y_center, width, height) in pixels
        │        │
        │        ├──► Y-center_px → KinematicMapper → steps_commanded
        │        │        H_mm   = (y_center_px / 224) × 300 mm
        │        │        steps  = round(H_mm × 80)          [80 steps/mm]
        │        │        clamp  → [0, 64000]                [0–800mm travel]
        │        │
        │        └──► Bounding Box Area (w × h pixels) → Spray Duration
        │                 A_small  (<2000 px²)  → 50ms   (low severity)
        │                 A_medium (2000–8000)  → 150ms  (moderate)
        │                 A_large  (>8000 px²)  → 350ms  (high severity)
        │
        ▼
JSON UART to Arduino: {"cmd":"GOTO","steps":N}
        │
        ▼
NEMA 17 Stepper moves nozzle to target height (~200 ms settle)
        │
        ▼
GPIO → Relay CH1: Nozzle Solenoid Opens → Spray → Closes
        │
        ▼
Log Entry Written: [timestamp, class, confidence, bbox, y_center_px, y_target_mm, steps_commanded, duration_ms]
```

### 4.3 Communication Protocol

| Link | Protocol | Baud / Speed | Payload Description |
|---|---|---|---|
| Pi ↔ Camera | USB 2.0 UVC | 30 FPS raw | 640×480 MJPEG frames |
| Pi ↔ Relay Module (nozzle + pump) | GPIO (BCM pins 17, 23) | N/A (direct) | 3.3V HIGH/LOW; CH1=nozzle, CH2=pump |
| Pi ↔ Arduino (stepper GOTO) | UART Serial | 115200 baud | JSON: `{"cmd":"GOTO","steps":N}` → `{"status":"ARRIVED"}` |
| Pi ↔ Arduino/ESP32 | UART Serial | 115200 baud | JSON packets: `{"cmd":"MOVE","spd":150,"dir":"FWD"}` |
| Arduino ↔ Motor Driver | PWM + Digital | 490 Hz PWM | ENA/ENB duty cycle + IN1-IN4 direction |
| Arduino ↔ Ultrasonics | Digital GPIO | Trigger/Echo | Distance in cm (2–400 cm range) |

### 4.4 Software Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Raspberry Pi 4 OS                    │
│                  (Raspberry Pi OS Lite 64-bit)        │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │             APPLICATION LAYER                   │  │
│  │                                                │  │
│  │  main_controller.py                            │  │
│  │  ├── CameraModule       (OpenCV VideoCapture)  │  │
│  │  ├── InferenceEngine    (PyTorch / ONNX RT)    │  │
│  │  ├── SprayDecisionEngine                       │  │
│  │  │   ├── KinematicMapper (Y-pixel → steps)     │  │
│  │  │   ├── StepperController (Serial GOTO/HOME)  │  │
│  │  │   ├── VRACalculator  (bbox area → ms)       │  │
│  │  │   └── RelayController(RPi.GPIO, 2-ch)       │  │
│  │  ├── NavigationBridge   (Serial → Arduino)     │  │
│  │  └── DataLogger         (CSV / SQLite)         │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │             MIDDLEWARE LAYER                     │  │
│  │  • OpenCV 4.x (image capture & pre-processing) │  │
│  │  • PyTorch 2.x / ONNX Runtime (inference)      │  │
│  │  • RPi.GPIO / gpiozero (relay actuation)        │  │
│  │  • pyserial (UART to Arduino)                   │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │             OS / DRIVER LAYER                   │  │
│  │  • V4L2 camera driver                          │  │
│  │  • BCM2711 GPIO driver                         │  │
│  │  • USB serial (FTDI/CP2102)                    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│               Arduino Nano / ESP32                    │
│                                                      │
│  motor_nav_controller.ino / rover_controller.ino             │
│  ├── MotorDriver      (L298N PWM control)            │
│  ├── UltrasonicSensor  (HC-SR04 × 2)                │
│  ├── RowFollower       (PID-based centering)         │
│  ├── ObstacleAvoider   (threshold-based stop/turn)   │
│  ├── StepperGantry     (AccelStepper, HOME/GOTO)     │
│  └── SerialListener    (parse JSON from Pi)          │
└──────────────────────────────────────────────────────┘
```

### 4.5 Model Quantization Strategy

The baseline ResNet-18 model (95.34% accuracy, ~44.7 MB FP32) must be optimized for Raspberry Pi 4 inference:

| Technique | Expected Size | Expected Accuracy | Inference Time (Pi 4) |
|---|---|---|---|
| FP32 (Baseline) | ~44.7 MB | 95.34% | ~800–1200 ms |
| FP16 (Half Precision) | ~22.4 MB | ~95.2% | ~500–700 ms |
| INT8 (Post-Training Quantization) | ~11.2 MB | ~94.0–94.8% | ~250–400 ms |
| ONNX Runtime + INT8 | ~11.2 MB | ~94.0–94.8% | ~200–350 ms |

**Selected Strategy:** ONNX export → ONNX Runtime with INT8 dynamic quantization. Fallback: PyTorch `torch.quantization.quantize_dynamic`.

### 4.6 Detection Architecture Adaptation

The current model is an **image-level classifier** (whole-image → class label). For field deployment with spatial targeting, the pipeline requires localization. Two approaches are evaluated:

| Approach | Complexity | Accuracy | Speed |
|---|---|---|---|
| **A: Sliding Window + Classifier** | Low | Moderate | Slow (multiple inferences/frame) |
| **B: GradCAM Heatmap → BBox Extraction** | Moderate | Good | Fast (single inference + post-processing) |
| **C: Fine-tune as Object Detector (YOLOv5-nano)** | High | Best | Fast (if model fits) |

**Selected Approach:** **Option B** for the MVP — run a single forward pass through the classifier, extract Grad-CAM activation maps from the final convolutional layer, threshold the heatmap, and compute a bounding box from the largest connected component. This provides:
- Disease class (from softmax output)
- Approximate spatial localization (from Grad-CAM heatmap)
- Severity proxy (thresholded heatmap area ≈ infection spread)

**Stretch Goal:** If time permits, fine-tune YOLOv5-nano for direct bounding-box detection.

---

## 5. Functional Requirements

### 5.1 User Stories

| ID | As a... | I want to... | So that... | Priority |
|---|---|---|---|---|
| US-01 | Farmer | Place the rover at the start of a tomato row and press a single START button | It autonomously traverses the row without further input | P0 (Must) |
| US-02 | Farmer | Have the rover detect diseased leaves automatically | I don't need to scout hundreds of plants manually | P0 (Must) |
| US-03 | Farmer | Have the rover spray only on infected areas at the correct height | I save fungicide and reduce chemical runoff | P0 (Must) |
| US-04 | Farmer | See an LED indicator showing the rover's status (idle / detecting / spraying / obstacle) | I know the system is working without a screen | P1 (Should) |
| US-05 | Farmer | Have the rover stop automatically when it detects an obstacle | The rover and my crops are not damaged | P0 (Must) |
| US-06 | Researcher | Access a CSV/JSON log of all detections with timestamps, confidence, zone, and spray duration | I can perform post-hoc analysis and publish results | P1 (Should) |
| US-07 | Researcher | Connect to the Pi via SSH/VNC during operation | I can monitor inference output in real time during field trials | P2 (Could) |
| US-08 | Extension Officer | Demonstrate the rover at a field day with minimal setup | Farmers see the technology in action and understand its value | P1 (Should) |

### 5.2 System Behavioral Requirements

#### FR-01: Image Capture
- **FR-01.1:** The system SHALL capture RGB frames at ≥2 FPS from a USB camera at 640×480 resolution.
- **FR-01.2:** The system SHALL skip frames when inference pipeline is backlogged (non-blocking capture).
- **FR-01.3:** The system SHALL auto-adjust exposure for outdoor daylight conditions (camera-level AE).

#### FR-02: Disease Detection
- **FR-02.1:** The system SHALL run the quantized ResNet-18 model on each captured frame.
- **FR-02.2:** The system SHALL classify each frame into one of the target classes: `{healthy, early_blight, late_blight}`. (Other classes from the 18-class model are treated as `other/ignore` in the field context.)
- **FR-02.3:** The system SHALL output a confidence score $P \in [0, 1]$ for the predicted class.
- **FR-02.4:** The system SHALL suppress spray actuation when confidence $P < 0.70$ (configurable threshold).
- **FR-02.5:** The system SHALL generate a Grad-CAM heatmap for each diseased prediction and extract a bounding box $(x, y, w, h)$ in pixel coordinates.

#### FR-03: Kinematic Nozzle Targeting
- **FR-03.1:** The system SHALL convert the bounding-box Y-center pixel ($y_{\text{px}} \in [0, 223]$) to a physical height in millimetres using the camera field-of-view constant:
$$
H_{\text{mm}} = \frac{y_{\text{px}}}{224} \times H_{\text{FOV}}
$$
  where $H_{\text{FOV}} = 300$ mm (configurable in `config.json`).
- **FR-03.2:** The system SHALL convert $H_{\text{mm}}$ to stepper steps using the kinematic constant of **80 steps/mm** (NEMA 17, 1/16 microstepping, GT2 2 mm pitch, 20-tooth pulley):
$$
\text{steps} = \text{round}\left( H_{\text{mm}} \times 80 \right), \quad \text{clamped to } [0,\, 64000]
$$
- **FR-03.3:** The system SHALL issue a `{"cmd":"GOTO","steps":N}` JSON command over UART to the Arduino before every spray event, and wait for `{"status":"ARRIVED"}` acknowledgement (timeout: 10 s).
- **FR-03.4:** The stepper gantry SHALL perform an automatic homing sequence (seek end-stop, zero position) on system startup and every 100 spray events.

#### FR-04: Variable Rate Application (VRA)
- **FR-04.1:** The system SHALL compute the spray relay-open duration as a linear function of the bounding-box pixel area:

$$
d_{\text{spray}} = \text{clamp}\left( \alpha \cdot A_{\text{bbox}} + \beta, \; d_{\min}, \; d_{\max} \right)
$$

where:
  - $A_{\text{bbox}} = w \times h$ (pixels²)
  - $\alpha, \beta$ are calibration constants (determined during commissioning)
  - $d_{\min} = 50 \text{ ms}$ (minimum actuation to overcome solenoid inertia)
  - $d_{\max} = 500 \text{ ms}$ (prevents over-application)

- **FR-04.2:** The VRA parameters SHALL be configurable via a JSON configuration file on the Pi's SD card.
- **FR-04.3:** The VRA parameters SHALL be configurable via a JSON configuration file on the Pi's SD card.

#### FR-05: Spray Actuation
- **FR-05.1:** The system SHALL control one 12V solenoid nozzle valve (GPIO 17) and one 12V diaphragm pump (GPIO 23) via a 2-channel relay module; the Z-axis stepper gantry positions the single nozzle at the computed target height before the relay fires.
- **FR-05.2:** The relay activation SHALL be non-blocking (timed using `asyncio` coroutines or a background thread).
- **FR-05.3:** The pump relay SHALL be energised concurrently with the nozzle relay and de-energised when spray completes.
- **FR-05.4:** The system SHALL enforce a minimum inter-spray cooldown of 200 ms per nozzle to prevent solenoid overheating.
- **FR-05.5:** The system SHALL log each spray event: `{timestamp, class, confidence, bbox_area, y_center_px, y_target_mm, steps_commanded, duration_ms}`.

#### FR-06: Autonomous Mobility
- **FR-06.1:** The Arduino/ESP32 SHALL accept movement commands from the Pi over UART serial at 115200 baud.
- **FR-06.2:** The default autonomous mode SHALL drive the rover forward at a configurable PWM speed (default: 60% duty cycle ≈ 0.2 m/s).
- **FR-06.3:** Two HC-SR04 ultrasonic sensors (left/right-front mounted) SHALL continuously measure distance to row boundaries.
- **FR-06.4:** A PID controller SHALL adjust left/right motor differential to maintain centering between rows.
- **FR-06.5:** If any ultrasonic sensor reads ≤ 15 cm (obstacle), the rover SHALL halt immediately and signal the Pi.
- **FR-06.6:** The rover SHALL support manual override via a physical DPDT kill switch that disconnects motor power.

#### FR-07: Data Logging
- **FR-07.1:** The system SHALL write a timestamped CSV log for every processed frame:
  ```
  timestamp, frame_id, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, y_center_px, y_target_mm, steps_commanded, spray_duration_ms, rover_speed
  ```
- **FR-07.2:** The system SHALL optionally save annotated frames (with bounding box and zone overlay) to disk at a configurable interval (default: every 10th frame).
- **FR-07.3:** Log files SHALL rotate at 50 MB to prevent SD card exhaustion.

#### FR-08: System Status Indication
- **FR-08.1:** An RGB LED SHALL indicate system state:
  - **Green (steady):** Idle / healthy detection, no spray.
  - **Red (blink):** Disease detected, spray in progress.
  - **Yellow (steady):** Obstacle detected, rover halted.
  - **Blue (blink):** System booting / model loading.

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Requirement ID | Requirement | Target |
|---|---|---|
| NFR-P01 | Inference latency (single frame, Pi 4 CPU) | ≤ 400 ms |
| NFR-P02 | End-to-end latency (capture → relay actuation) | ≤ 750 ms |
| NFR-P03 | Minimum sustained frame rate | ≥ 2 FPS |
| NFR-P04 | Rover ground speed during detection | 0.1–0.3 m/s |
| NFR-P05 | Time from power-on to operational state | ≤ 60 seconds |

### 6.2 Reliability

| Requirement ID | Requirement | Target |
|---|---|---|
| NFR-R01 | Continuous operation without crash/hang | ≥ 45 minutes |
| NFR-R02 | Relay actuation reliability | ≥ 99% (1000-cycle test) |
| NFR-R03 | Graceful handling of camera disconnection | System pauses, LED yellow, retry 3× |
| NFR-R04 | Watchdog timer on Arduino | Auto-stop motors if no serial heartbeat for 2s |
| NFR-R05 | SD card write reliability | Use write-ahead journaling; fsync after each log batch |

### 6.3 Environmental & Safety

| Requirement ID | Requirement | Specification |
|---|---|---|
| NFR-E01 | Operating temperature range | 15°C – 45°C |
| NFR-E02 | Ingress protection (spray-side electronics) | IP44 equivalent (splash-proof enclosure for Pi + relays) |
| NFR-E03 | Battery chemistry safety | LiPo with BMS; OR sealed lead-acid (SLA) 12V 7Ah |
| NFR-E04 | Emergency stop | Physical kill switch disconnects battery from motor driver |
| NFR-E05 | Chemical compatibility | All fluid-contact components (PVC, nozzles, tubing) rated for Mancozeb solution |
| NFR-E06 | Rover weight (loaded) | ≤ 8 kg (to prevent soil compaction in wet field) |

### 6.4 Maintainability & Extensibility

| Requirement ID | Requirement | Specification |
|---|---|---|
| NFR-M01 | Modular software architecture | Separate modules for camera, inference, spraying, navigation |
| NFR-M02 | Configuration-driven parameters | All thresholds, kinematic constants, VRA params in a single `config.json` |
| NFR-M03 | OTA-capable model updates | Drop new `.onnx` model on SD card; system loads on next boot |
| NFR-M04 | Standard connectors | JST-XH for all inter-board wiring; quick-disconnect for spray lines |
| NFR-M05 | Code documentation | All Python modules with Google-style docstrings; Arduino with Doxygen comments |

### 6.5 Power Budget

| Component | Voltage | Current (typ.) | Power (W) |
|---|---|---|---|
| Raspberry Pi 4 (4 GB) | 5V | 1.5 A | 7.5 W |
| USB Camera | 5V (via Pi USB) | 0.2 A | 1.0 W |
| Arduino Nano / ESP32 | 5V | 0.05 A | 0.25 W |
| 4× DC Geared Motors | 12V | 0.6 A total | 7.2 W |
| L298N / BTS7960 Driver | 12V | 0.1 A quiescent | 1.2 W |
| NEMA 17 Stepper Motor | 12V | 0.8 A (peak) | 9.6 W peak |
| A4988/DRV8825 Stepper Driver | 12V | 0.05 A quiescent | 0.6 W |
| 1× Solenoid Valve (intermittent) | 12V | 0.3 A (peak) | 3.6 W peak |
| 12V Diaphragm Pump (intermittent) | 12V | 1.5 A (peak) | 18.0 W peak |
| 2-Channel Relay Module | 5V | 0.04 A | 0.2 W |
| Status LEDs + Misc | 5V | 0.05 A | 0.25 W |
| **Continuous Total** | — | — | **~19.2 W** |
| **Peak Total (spraying + stepper)** | — | — | **~41.2 W** |

**Battery Specification:** 12V 7Ah Sealed Lead-Acid (SLA) = 84 Wh → **~4.4 hours at continuous draw** (theoretical); **~2.0 hours at peak-averaged draw** (safety margin). A step-down buck converter (12V→5V, 3A) powers the Pi and logic boards.

---

## 7. Bill of Materials & Estimated Budget Breakdown

### 7.1 Detailed Bill of Materials

| # | Component | Specification | Qty | Unit Cost (INR) | Total (INR) | Source |
|---|---|---|---|---|---|---|
| **Compute & Sensing** | | | | | | |
| 1 | Raspberry Pi 4 Model B | 4 GB RAM, BCM2711 | 1 | 4,500 | 4,500 | Robu.in / Amazon.in |
| 2 | MicroSD Card | 32 GB Class 10 UHS-I | 1 | 350 | 350 | Amazon.in |
| 3 | USB Camera | Logitech C270 720p / Generic 5MP USB | 1 | 800 | 800 | Amazon.in |
| 4 | Pi Power Supply / Buck Converter | LM2596 12V→5V 3A step-down | 1 | 120 | 120 | Robu.in |
| **Mobility** | | | | | | |
| 5 | 4WD Robot Chassis Kit | Acrylic plate + 4× BO motors + wheels | 1 | 1,200 | 1,200 | Robu.in |
| 6 | L298N Motor Driver Module | Dual H-Bridge, 2A per channel | 1 | 180 | 180 | Robu.in |
| 7 | Arduino Nano (clone) | ATmega328P, CH340 USB | 1 | 250 | 250 | Robu.in |
| 8 | HC-SR04 Ultrasonic Sensor | 2–400 cm range | 2 | 50 | 100 | Robu.in |
| **Spray Actuation** | | | | | | |
| 9 | 12V Solenoid Valve (normally closed) | 1/4" BSP, 0–0.8 MPa | 1 | 350 | 350 | Amazon.in |
| 10 | 12V Diaphragm Pump | 3–5 LPM, self-priming | 1 | 450 | 450 | Amazon.in |
| 11 | 2-Channel 5V Relay Module | Optocoupler-isolated, 10A/250VAC | 1 | 100 | 100 | Robu.in |
| 12 | NEMA 17 Stepper Motor | 1.8°/step, 200 steps/rev, 1.5A | 1 | 400 | 400 | Robu.in |
| 13 | A4988 / DRV8825 Stepper Driver | 1/16 microstepping, up to 2A | 1 | 150 | 150 | Robu.in |
| 14 | 2040 V-Slot Aluminium Rail | 800 mm, anodised | 1 | 600 | 600 | Robu.in / AliExpress |
| 15 | GT2 Timing Belt + 20T Pulley | 1 m belt, 2 mm pitch, 8 mm bore | 1 set | 200 | 200 | Robu.in |
| 16 | End-Stop Microswitch | Lever-type, 5V logic | 1 | 50 | 50 | Robu.in |
| 17 | Flat-Fan Spray Nozzle | 80° fan, 0.3 GPM | 1 | 80 | 80 | Amazon.in |
| 18 | Silicone Tubing (6mm ID) | 3 meters | 1 | 150 | 150 | Amazon.in |
| 19 | Spray Reservoir / Tank | 2L plastic bottle with cap fitting | 1 | 50 | 50 | Local |
| **Power** | | | | | | |
| 20 | 12V 7Ah SLA Battery | Sealed lead-acid, rechargeable | 1 | 1,100 | 1,100 | Amazon.in |
| 21 | Battery Charger (12V SLA) | Float charger, 1A | 1 | 350 | 350 | Amazon.in |
| **Wiring & Miscellaneous** | | | | | | |
| 22 | Jumper Wires (M-M, M-F, F-F) | 40-pin ribbon × 3 types | 3 | 60 | 180 | Robu.in |
| 23 | Breadboard (830 tie-point) | Full-size | 1 | 100 | 100 | Robu.in |
| 24 | Toggle / Kill Switch | DPDT, 10A rated | 1 | 40 | 40 | Local |
| 25 | Status LEDs (RGB) + Resistors | 5mm common-cathode + 220Ω | 5 | 5 | 25 | Robu.in |
| 26 | Cable Ties, Heat Shrink, Electrical Tape | Assorted | 1 set | 100 | 100 | Local |
| 27 | Acrylic / 3D-Printed Mounts | Camera mount, Pi enclosure, gantry bracket | 1 set | 300 | 300 | University FabLab |
| 28 | Flyback Diodes (1N4007) | For solenoid back-EMF protection | 2 | 3 | 6 | Robu.in |
| 29 | MOSFET / Transistor (for relay drive) | 2N2222 / IRLZ44N | 2 | 10 | 20 | Robu.in |

### 7.2 Budget Summary

| Category | Subtotal (INR) |
|---|---|
| Compute & Sensing | 5,770 |
| Mobility | 1,730 |
| Spray Actuation | 1,780 |
| Power | 1,450 |
| Wiring & Miscellaneous | 771 |
| **Subtotal** | **11,501** |
| Shipping & Handling (~10%) | 1,150 |
| Contingency & Replacements (~15%) | 1,725 |
| **Grand Total (Estimated)** | **≈ 14,376** |

> **Budget Margin:** INR 18,000 – INR 14,376 = **INR 3,624 remaining** (~20.1% buffer for unforeseen expenses, replacement parts, or optional upgrades such as a Pi Camera Module v2 or BTS7960 high-current driver).

### 7.3 Optional Upgrades (within budget margin)

| Upgrade | Cost (INR) | Benefit |
|---|---|---|
| Pi Camera Module v2 (8MP, CSI) | ~1,200 | Reduced USB latency, better low-light |
| BTS7960 43A Motor Driver | ~350 | Higher current headroom for larger motors |
| 0.96" I2C OLED Display | ~200 | On-rover status display (class, confidence, zone) |
| GPS Module (NEO-6M) | ~350 | Geotag spray events for field mapping |

---

## 8. Development Timeline (12-Week Phased Approach)

### 8.1 Phase Overview

```
        Week 1-2        Week 3-4        Week 5-6        Week 7-8        Week 9-10       Week 11-12
       ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
       │ PHASE 1  │   │ PHASE 2  │   │ PHASE 3  │   │ PHASE 4  │   │ PHASE 5  │   │ PHASE 6  │
       │ Procure  │──▶│ Edge AI  │──▶│ Spray    │──▶│ Mobility │──▶│ System   │──▶│ Field    │
       │ & Setup  │   │ Pipeline │   │ Actuation│   │ + Nav    │   │ Integr.  │   │ Trials & │
       │          │   │          │   │          │   │          │   │          │   │ Write-Up │
       └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### 8.2 Detailed Weekly Plan

#### Phase 1: Procurement, Environment Setup & Model Quantization (Weeks 1–2)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W1 | Finalize BoM; place orders on Robu.in / Amazon.in | Chirag, Sayed | Purchase orders, tracking IDs |
| W1 | Set up Raspberry Pi OS (64-bit Lite), install PyTorch, ONNX Runtime, OpenCV | Vinayak | Bootable Pi with inference stack |
| W1 | Export trained ResNet-18 (Epoch 30) to ONNX format; apply INT8 post-training quantization | Vinayak, Omkar | `resnet18_crop_int8.onnx` (≤12 MB) |
| W2 | Benchmark quantized model on Pi 4: measure FPS, accuracy degradation | Vinayak | Benchmark report: FPS, accuracy |
| W2 | Implement Grad-CAM extraction pipeline on Pi (OpenCV + NumPy) | Omkar | `gradcam.py` module; sample heatmap outputs |
| W2 | Assemble 4WD chassis; verify motor/driver connectivity | Chirag, Sayed | Rolling chassis with motor test video |

**Phase 1 Gate:** Quantized model runs on Pi ≥2 FPS with ≤2% accuracy drop. Chassis moves forward/reverse.

#### Phase 2: Edge-AI Pipeline & Camera Integration (Weeks 3–4)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W3 | Develop `CameraModule` (OpenCV capture, auto-exposure, frame skip) | Omkar | `camera_module.py` |
| W3 | Develop `InferenceEngine` (ONNX RT session, pre-process, post-process, softmax) | Vinayak | `inference_engine.py` |
| W3 | Implement bounding-box extraction from Grad-CAM heatmap | Omkar | `bbox_extractor.py` |
| W4 | Develop `SprayDecisionEngine`: zone mapper + VRA calculator | Vinayak | `spray_decision.py` |
| W4 | Unit tests: feed 100 lab images through full pipeline on Pi; verify class, bbox, zone, duration outputs | All | Unit test report (≥95% pass rate) |
| W4 | Design and print/cut camera mount and Pi enclosure (IPx4 splash-proof) | Chirag | Physical mount + enclosure |

**Phase 2 Gate:** Full software pipeline (capture → inference → zone → duration) running end-to-end on Pi with logged CSV output.

#### Phase 3: Spray Actuation System — Z-Axis Stepper Gantry (Weeks 5–6)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W5 | Assemble 2040 V-slot rail (80 cm) with GT2 belt, 20-tooth pulley, and stepper motor mount | Chirag, Sayed | Physical gantry assembly (photos) |
| W5 | Wire NEMA 17 → A4988/DRV8825 driver → Arduino (STEP D2, DIR D3, EN D4); install end-stop on D5 | Sayed | Wiring diagram + stepper jog test video |
| W5 | Flash `rover_controller.ino` with AccelStepper; verify HOME/GOTO/STOP JSON commands over serial | Vinayak | Serial command test log |
| W6 | Mount single nozzle carriage on gantry; plumb pump → tubing → nozzle; pressure test | Chirag | Leak-free spray system (video proof) |
| W6 | 5-point kinematic calibration: command 0, 200, 400, 600, 800 mm; measure actual nozzle height with ruler; verify ≤±15 mm | All | Calibration report with positional error measurements |
| W6 | Integration test: Pi detects disease in image → `stepper.goto(steps)` → `relay.spray()` fires at correct height | All | Integration test video + log file |

**Phase 3 Gate:** Stepper gantry homes reliably; 5-point kinematic accuracy ≤±15 mm; single nozzle sprays at correct height when shown disease images.

#### Phase 4: Autonomous Mobility & Navigation (Weeks 7–8)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W7 | Flash Arduino Nano; implement motor control (PWM speed, direction) | Sayed | `motor_nav_controller.ino` |
| W7 | Implement ultrasonic sensor reading + distance averaging (median filter, 5 samples) | Sayed | Sensor test log (accuracy ±2 cm) |
| W7 | Implement PID row-following: `error = dist_left – dist_right; adjust PWM` | Sayed, Omkar | PID-tuned straight-line test (video) |
| W8 | Implement obstacle avoidance: stop if any sensor ≤ 15 cm; back up + signal Pi | Sayed | Obstacle course test (10/10 stops) |
| W8 | Develop UART serial protocol between Pi and Arduino (JSON commands) | Vinayak | `NavigationBridge` on Pi; serial parser on Arduino |
| W8 | Implement watchdog: Arduino auto-stops motors if no heartbeat from Pi for 2 seconds | Sayed | Watchdog test (disconnect USB, motors stop) |

**Phase 4 Gate:** Rover follows a straight corridor, stops for obstacles, responds to Pi serial commands.

#### Phase 5: Full System Integration & Lab Testing (Weeks 9–10)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W9 | Mount all subsystems on chassis (Pi, camera, boom, battery, pump, tank) | Chirag, Sayed | Fully assembled rover (photos, weight measurement) |
| W9 | Develop `main_controller.py` (main loop: capture → infer → spray → log → move) | Vinayak | Main controller with state machine |
| W9 | Implement `config.json` (all thresholds, zone boundaries, VRA params, speeds) | Omkar | Configuration file + schema documentation |
| W10 | Indoor integration test: place printed disease images on a mock plant stand; rover drives past, detects, sprays correct zone | All | Integration test video + CSV log analysis |
| W10 | Stress test: 45-minute continuous run; monitor for crashes, memory leaks, overheating | All | Stability report (uptime, CPU temp, RAM usage) |
| W10 | Implement status LED state machine | Omkar | LED test confirmation |

**Phase 5 Gate:** Rover completes a full indoor dry run: drives, detects, sprays correct zones, logs data, runs ≥45 min.

#### Phase 6: Field Validation, Data Collection & Documentation (Weeks 11–12)

| Week | Task | Owner | Deliverable |
|---|---|---|---|
| W11 | Field trial #1 at REVA University agricultural plot: deploy on live tomato rows | All | Raw field data (CSV logs, annotated frames) |
| W11 | Ground-truth annotation: manually label 200+ field frames for disease presence/absence and zone correctness | Omkar, Chirag | Ground-truth dataset |
| W11 | Compute field metrics: accuracy, precision, recall, F1, zone accuracy, spray proportionality R² | Vinayak | Field validation report |
| W11 | Field trial #2: volumetric comparison (rover-targeted spray vs. blanket spray on adjacent rows) | All | Fungicide usage comparison data |
| W12 | Write IEEE-format conference paper (IMRaD structure) | All | Draft paper (≥6 pages, IEEE double-column) |
| W12 | Prepare project report, poster, and demo video for academic panel | All | Final report, poster PDF, 3-min demo video |
| W12 | Final presentation and live demo to evaluation panel | All | Panel evaluation score |

**Phase 6 Gate:** ≥85% field detection accuracy, ≥85% zone accuracy, ≥30% fungicide reduction demonstrated, paper draft complete.

### 8.3 Gantt Chart (Simplified)

```
Week:     1    2    3    4    5    6    7    8    9    10   11   12
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Phase 1:  ████████ │    │    │    │    │    │    │    │    │    │
Phase 2:  │    │ ████████ │    │    │    │    │    │    │    │    │
Phase 3:  │    │    │    │████████ │    │    │    │    │    │    │
Phase 4:  │    │    │    │    │    │████████ │    │    │    │    │
Phase 5:  │    │    │    │    │    │    │    │████████ │    │    │
Phase 6:  │    │    │    │    │    │    │    │    │    │████████ │
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Reviews:       G1        G2        G3        G4        G5   FINAL
```

---

## 9. Academic Validation & Testing Plan

### 9.1 Validation Framework

The system is evaluated across three complementary validation dimensions:

```
┌─────────────────────────────────────────────────────┐
│              VALIDATION PYRAMID                      │
│                                                     │
│                    ┌────────┐                        │
│                    │ FIELD  │  Cross-domain          │
│                    │ TRIAL  │  validation (real      │
│                    │        │  plants, natural       │
│                    │  N≥200 │  sunlight)             │
│                   ┌┴────────┴┐                       │
│                   │  LAB     │  Controlled indoor    │
│                   │  BENCH   │  test (printed        │
│                   │  TEST    │  images, mock         │
│                   │  N≥100   │  plants)              │
│                  ┌┴──────────┴┐                      │
│                  │  OFFLINE   │  Test-set evaluation  │
│                  │  MODEL     │  (20,042 images,     │
│                  │  EVAL      │  cleaned dataset)     │
│                  │  N=20,042  │                       │
│                  └────────────┘                       │
└─────────────────────────────────────────────────────┘
```

### 9.2 Test Plan Overview

| Test ID | Test Name | Phase | Environment | Sample Size | Pass Criteria |
|---|---|---|---|---|---|
| T-01 | Offline Model Accuracy | Phase 2 | Desktop / Pi | 20,042 test images | Accuracy ≥ 95% (baseline: 95.34%) |
| T-02 | Quantized Model Accuracy | Phase 1 | Pi 4 | 20,042 test images | Accuracy ≥ 93% (≤2% degradation) |
| T-03 | Edge Inference Latency | Phase 2 | Pi 4 | 500 frames | Mean ≤ 400 ms, P95 ≤ 600 ms |
| T-04 | Grad-CAM BBox Quality | Phase 2 | Pi 4 | 200 images | IoU ≥ 0.40 vs. manual annotation |
| T-05 | Kinematic Targeting Accuracy | Phase 3 | Lab bench | 5 target heights (0, 200, 400, 600, 800 mm) | Positional error ≤ ±15 mm at each height |
| T-06 | VRA Duration Linearity | Phase 3 | Lab bench | 50 area-duration pairs | R² ≥ 0.80 |
| T-07 | Relay Actuation Reliability | Phase 3 | Lab bench | 1000 ON/OFF cycles per valve | ≥ 99% successful actuations |
| T-08 | Row-Following Accuracy | Phase 4 | Indoor corridor | 10 × 5m runs | Lateral deviation ≤ ±10 cm |
| T-09 | Obstacle Avoidance | Phase 4 | Indoor corridor | 50 obstacle placements | ≥ 90% successful stops |
| T-10 | System Uptime | Phase 5 | Lab | 3 × 45-min runs | Zero crashes; CPU temp ≤ 80°C |
| T-11 | End-to-End Indoor | Phase 5 | Lab mock-up | 50 disease presentations | ≥ 75% correct detect + kinematic position + spray |
| T-12 | Field Detection Accuracy | Phase 6 | REVA Agri Plot | ≥ 200 field frames | Accuracy ≥ 85% |
| T-13 | Field Kinematic Accuracy | Phase 6 | REVA Agri Plot | ≥ 50 spray events | Nozzle height within ±30 mm of leaf centre (water-sensitive paper) |
| T-14 | Fungicide Reduction | Phase 6 | REVA Agri Plot | 2 adjacent rows | ≥ 30% volume reduction vs. broadcast |
| T-15 | Field Latency | Phase 6 | REVA Agri Plot | 100 frames | End-to-end ≤ 750 ms |

### 9.3 Cross-Domain Validation Protocol (Lab → Field)

To quantify the **domain shift** between the training dataset (curated internet-sourced images) and real-world field conditions, the following protocol is executed:

#### Step 1: Baseline Establishment
- Run the quantized model on the full 20,042-image test set.
- Record per-class precision, recall, and F1-score.
- This is the **Lab Baseline** ($B_{\text{lab}}$).

#### Step 2: Field Image Collection
- Deploy the rover in REVA University's tomato plot during peak blight season.
- Capture ≥200 frames with disease present and ≥100 healthy frames.
- Collect at multiple times of day (09:00, 12:00, 15:00) to capture lighting variation.

#### Step 3: Ground-Truth Annotation
- Two team members independently label each field frame:
  - Disease class (healthy / early blight / late blight)
  - Severity (low / moderate / high — visual proxy)
  - Zone where disease is most prominent (top / mid / bottom as reference only)
- Inter-annotator agreement measured via Cohen's Kappa ($\kappa$).
  - Target: $\kappa \geq 0.80$ (substantial agreement).
- Disagreements resolved by third-member adjudication.

#### Step 4: Field Metric Computation
- Run the rover's inference pipeline on all field-collected frames.
- Compute:
  - **Field Accuracy** ($A_{\text{field}}$): Overall classification correctness.
  - **Field Precision / Recall / F1**: Per-class, weighted average.
  - **Domain Shift Gap**: $\Delta = B_{\text{lab}} - A_{\text{field}}$.
  - **Confusion Matrix**: Visualise systematic misclassifications.
  - **Kinematic Accuracy**: Mean and max positional error (mm) per spray event (water-sensitive paper).

#### Step 5: Spray System Validation
- For each correctly detected disease frame in the field:
  - Record which nozzle(s) activated.
  - Record spray duration.
  - Compare against ground-truth zone annotation → compute **Zone Accuracy**.
- Measure total spray volume for rover-targeted rows vs. manually broadcast-sprayed control rows → compute **Fungicide Reduction %**.

### 9.4 Statistical Analysis Plan

| Analysis | Method | Tool |
|---|---|---|
| Classification Performance | Precision, Recall, F1, Accuracy, Confusion Matrix | scikit-learn |
| Domain Shift Quantification | Paired comparison of per-class F1 (lab vs. field) | Wilcoxon signed-rank test |
| VRA Linearity | Linear regression (bbox area vs. spray duration) | scipy.stats.linregress |
| Inter-Annotator Agreement | Cohen's Kappa | scikit-learn `cohen_kappa_score` |
| System Reliability | Mean Time Between Failures (MTBF) from uptime logs | Manual computation |
| Fungicide Savings | Two-sample t-test on volumetric measurements | scipy.stats.ttest_ind |

### 9.5 Publication-Ready Deliverables

| Deliverable | Format | Target Venue |
|---|---|---|
| IEEE Conference Paper | 6–8 pages, IEEE double-column (IEEEtran template) | IEEE International Conference on Precision Agriculture Technology / ICPA |
| Technical Report | University capstone report format (~40 pages) | REVA University CSE Department |
| Demo Video | 3-minute narrated video (field trial footage) | Conference supplementary / YouTube |
| Open-Source Repository | GitHub (code, CAD files, BoM, documentation) | Public access after submission |
| Poster | A0 portrait, IEEE-style layout | University project expo |

### 9.6 Evaluation Metrics Summary Table (for Paper)

The following table is designed for direct inclusion in the IEEE paper's Results section:

| Metric | Lab Baseline | Field Result | Δ (Gap) | Pass? |
|---|---|---|---|---|
| Overall Accuracy | 95.34% | TBD | TBD | ≥85% |
| Blight Precision | TBD | TBD | TBD | ≥85% |
| Blight Recall | TBD | TBD | TBD | ≥85% |
| Edge FPS | TBD | TBD | N/A | ≥2 FPS |
| E2E Latency | TBD | TBD | N/A | ≤750 ms |
| Kinematic Accuracy | N/A | TBD | N/A | ≤±15 mm (lab) / ≤±30 mm (field) |
| VRA R² | N/A | TBD | N/A | ≥0.80 |
| Fungicide Reduction | N/A | TBD | N/A | ≥30% |

---

## 10. Risk Register & Mitigation

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy |
|---|---|---|---|---|
| R-01 | **Quantized model accuracy drops >5% on Pi** | Medium | High | Pre-validate with calibration dataset; fallback to FP16 (larger but more accurate) |
| R-02 | **Inference latency exceeds 500 ms** | Medium | High | Use ONNX Runtime instead of PyTorch; reduce input resolution to 160×160; profile and optimize pre-processing |
| R-03 | **Domain shift causes ≤70% field accuracy** | High | High | Collect 50+ field images early (Week 3); fine-tune model with augmented field data; adjust confidence threshold |
| R-04 | **Stepper gantry loses steps / skips under load** | Low | Medium | Enable AccelStepper acceleration/deceleration; verify motor current limit on A4988; re-home every 100 cycles |
| R-04b | **Nozzle carriage jams on V-slot rail (debris, misalignment)** | Low | Medium | Wipe rail before field deployment; use V-wheel eccentric nut adjustment; physical end-stop prevents over-travel |
| R-05 | **Pi overheats in direct sunlight (throttling)** | Medium | Medium | Mount small heatsink + fan on SoC; shade enclosure; throttle alert at 75°C |
| R-06 | **Battery insufficient for 45-min field trial** | Low | Medium | Pre-test with full charge; carry backup battery; optimize motor PWM |
| R-07 | **Ultrasonic sensors unreliable in dense canopy** | Medium | Low | Use median-filter (5 readings); reduce speed; fall back to timed straight-line drive |
| R-08 | **Component delivery delays** | Medium | Medium | Order in Week 1 with express shipping; identify local Bengaluru electronics markets (SP Road) as backup |
| R-09 | **Water ingress damages electronics** | Low | High | IP44 enclosure for Pi/relays; conformal coating on exposed PCBs; test spray pattern away from electronics |
| R-10 | **Team member unavailability** | Low | Medium | Cross-train on all subsystems; shared GitHub repo with CI; each member has a backup for their module |

---

## 11. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
2. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint arXiv:1704.04861*.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.
4. Gil, E., et al. (2014). Variable rate sprayer. Part 2—Vineyard prototype: Design, implementation, and validation. *Computers and Electronics in Agriculture*, 95, 136–150.
5. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.
6. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
7. Barbedo, J. G. A. (2019). Plant disease identification from individual lesions and spots using deep learning. *Biosystems Engineering*, 180, 96–107.
8. National Horticulture Board (NHB), India. (2023). *Indian Horticulture Database*.
9. FRAC (Fungicide Resistance Action Committee). (2022). *FRAC Code List: Fungal control agents sorted by cross-resistance pattern and mode of action*.
10. Agricultural Census Division, Ministry of Agriculture & Farmers Welfare, Government of India. (2015-16). *All India Report on Number and Area of Operational Holdings*.
11. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv preprint arXiv:1804.02767*.
12. PyTorch Documentation. (2025). *Post-Training Static Quantization*. https://pytorch.org/docs/stable/quantization.html.
13. ONNX Runtime Documentation. (2025). *Quantization*. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html.

---

## 12. Revision History

| Version | Date | Author(s) | Changes |
|---|---|---|---|
| 1.0 | 23 Feb 2026 | Chirag, Sayed, Omkar, Vinayak | Initial PRD release |
| 2.0 | 23 Feb 2026 | Chirag, Sayed, Omkar, Vinayak | Architecture baseline: 3-zone solenoid boom, 4-ch relay, software ZoneMapper |
| 3.0 | 7 Mar 2026 | Chirag, Sayed, Omkar, Vinayak | **Architectural pivot:** replaced 3-zone discrete solenoid boom with 1-DOF Z-axis linear actuator (NEMA 17 + GT2 belt + 2040 V-slot 80 cm); replaced ZoneMapper with KinematicMapper + StepperController; updated FR-03, FR-05, BoM, VRA log schema, power budget, Phase 3 timeline, validation tests, risk register, and all appendices |

---

## Appendix A: Configuration File Schema

```json
{
  "model": {
    "path": "models/resnet18_crop_int8.onnx",
    "input_size": [224, 224],
    "confidence_threshold": 0.70,
    "target_classes": ["early_blight", "late_blight"],
    "healthy_class": "healthy"
  },
  "camera": {
    "device_id": 0,
    "resolution": [640, 480],
    "fps": 30,
    "auto_exposure": true
  },
  "stepper": {
    "steps_per_mm": 80,
    "travel_mm": 800,
    "home_offset_mm": 20,
    "max_speed_steps_per_sec": 4000,
    "acceleration": 2000,
    "nozzle_gpio_pin": 17,
    "pump_gpio_pin": 23,
    "camera_fov_height_mm": 300,
    "homing_timeout_s": 15,
    "goto_timeout_s": 10
  },
  "vra": {
    "alpha": 0.025,
    "beta": 30.0,
    "min_duration_ms": 50,
    "max_duration_ms": 500,
    "cooldown_ms": 200
  },
  "navigation": {
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 115200,
    "default_speed_pwm": 150,
    "obstacle_threshold_cm": 15,
    "heartbeat_interval_ms": 500
  },
  "logging": {
    "csv_path": "logs/spray_log.csv",
    "save_frames": true,
    "frame_save_interval": 10,
    "max_log_size_mb": 50
  }
}
```

## Appendix B: Serial Communication Protocol (Pi ↔ Arduino)

### Pi → Arduino (JSON Commands)

```json
{"cmd": "MOVE", "dir": "FWD", "spd": 150}
{"cmd": "MOVE", "dir": "REV", "spd": 100}
{"cmd": "MOVE", "dir": "LEFT", "spd": 120}
{"cmd": "MOVE", "dir": "RIGHT", "spd": 120}
{"cmd": "STOP"}
{"cmd": "HEARTBEAT"}
{"cmd": "HOME"}
{"cmd": "GOTO", "steps": 12000}
{"cmd": "POS"}
{"cmd": "PING"}
```

### Arduino → Pi (JSON Status)

```json
{"status": "OK", "dist_l": 45.2, "dist_r": 42.8, "obstacle": false}
{"status": "OBSTACLE", "dist_l": 12.4, "dist_r": 45.1, "obstacle": true}
{"status": "WATCHDOG", "msg": "No heartbeat for 2000ms, motors stopped"}
{"status": "HOMED", "steps": 0}
{"status": "ARRIVED", "steps": 12000}
{"status": "STOPPED"}
{"status": "POS", "steps": 12000}
{"status": "PONG"}
```

## Appendix C: Wiring Pinout Summary

### Raspberry Pi 4 GPIO Allocation

| GPIO (BCM) | Function | Connected To |
|---|---|---|
| GPIO 17 | Relay CH1 (Nozzle solenoid) | Relay module IN1 |
| GPIO 23 | Relay CH2 (Pump) | Relay module IN2 |
| GPIO 24 | Status LED - Red | 220Ω → LED → GND |
| GPIO 25 | Status LED - Green | 220Ω → LED → GND |
| GPIO 12 | Status LED - Blue | 220Ω → LED → GND |
| TXD (GPIO 14) | UART TX to Arduino | Arduino RX (via level shifter) |
| RXD (GPIO 15) | UART RX from Arduino | Arduino TX (via level shifter) |

### Arduino Nano Pin Allocation

| Pin | Function | Connected To |
|---|---|---|
| D2 | Stepper STEP | A4988/DRV8825 STEP |
| D3 | Stepper DIR | A4988/DRV8825 DIR |
| D4 | Stepper ENABLE (active LOW) | A4988/DRV8825 EN |
| D5 | End-stop input (INPUT_PULLUP) | Limit microswitch |
| D6 (PWM) | Motor A Enable (ENA) | L298N ENA |
| D7 | Motor A IN1 | L298N IN1 |
| D8 | Motor A IN2 | L298N IN2 |
| D9 (PWM) | Motor B Enable (ENB) | L298N ENB |
| D10 | Motor B IN3 | L298N IN3 |
| D11 | Motor B IN4 | L298N IN4 |
| A0 | Ultrasonic L - Trigger | HC-SR04 #1 Trig |
| A1 | Ultrasonic L - Echo | HC-SR04 #1 Echo |
| A2 | Ultrasonic R - Trigger | HC-SR04 #2 Trig |
| A3 | Ultrasonic R - Echo | HC-SR04 #2 Echo |
| RX (D0) | UART RX from Pi | Pi TXD (via level shifter) |
| TX (D1) | UART TX to Pi | Pi RXD (via level shifter) |

---

*End of Document*

**Prepared by:** Team Chirag, Sayed, Omkar, Vinayak  
**Department of Computer Science and Engineering**  
**REVA University, Bengaluru — 560064**  
**Academic Year 2025–26**
