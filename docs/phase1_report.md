# Phase 1: BDD100K Data Analysis Report

**Project:** Object Detection Data Analysis -- Bosch Applied CV Assignment
**Dataset:** BDD100K (Berkeley DeepDrive)
**Date:** 2026-02-23
**Scope:** 10 detection classes, train and validation splits only

---

## 1. Executive Summary

This report presents a quantitative analysis of the BDD100K dataset for the object detection task across 10 classes. The dataset contains **1,472,397 bounding box annotations** spanning **79,863 unique images**. The analysis reveals three dominant characteristics that will directly impact model design:

1. **Severe class imbalance** -- `car` accounts for 55.4% of all annotations while `train` represents just 0.01%, a ratio exceeding 5,000:1.
2. **Small object prevalence** -- 87.4% of all bounding boxes occupy less than 1% of image area, making small-object detection a first-order concern.
3. **High occlusion rates for vulnerable road users** -- `rider` (89.1%) and `bike` (84.3%) are overwhelmingly occluded, posing a safety-critical detection challenge.

The train/val split is statistically well-balanced (chi-squared p=0.198), confirming that class proportions are preserved across splits and validation metrics will be representative.

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| Total annotations | 1,472,397 |
| Unique images | 79,863 |
| Train annotations | 1,286,871 (87.4%) |
| Val annotations | 185,526 (12.6%) |
| Detection classes | 10 |
| Image resolution | 1280 x 720 (constant) |
| Empty images | 0 |

The train/val annotation ratio (~87/13) aligns with the standard BDD100K split of 70K/10K images. Every image in the dataset contains at least 3 annotated objects, confirming no degenerate samples exist.

---

## 3. Class Distribution Analysis

| Class | Count | Percentage |
|---|---|---|
| car | 815,717 | 55.40% |
| traffic sign | 274,594 | 18.65% |
| traffic light | 213,002 | 14.47% |
| person | 104,611 | 7.11% |
| truck | 34,216 | 2.32% |
| bus | 13,269 | 0.90% |
| bike | 8,217 | 0.56% |
| rider | 5,166 | 0.35% |
| motor | 3,454 | 0.23% |
| train | 151 | 0.01% |

**Imbalance characterization.** The distribution follows a heavy long-tail pattern. Three classes (`car`, `traffic sign`, `traffic light`) account for 88.5% of all annotations. The remaining seven classes share 11.5%. The head-to-tail ratio (`car` to `train`) is 5,402:1.

This imbalance is not a dataset defect -- it reflects real-world driving frequency. However, it has direct consequences for model training:

- A naive cross-entropy loss will bias the detector heavily toward `car`.
- Rare classes (`train`, `motor`, `rider`) will be under-represented in gradient updates.
- Evaluation metrics must be examined per-class; aggregate mAP alone will mask poor performance on tail classes.

---

## 4. Train/Val Split Analysis

A chi-squared goodness-of-fit test was used to evaluate whether the class distributions in the train and validation splits are drawn from the same underlying distribution.

| Metric | Value |
|---|---|
| Chi-squared statistic | 12.27 |
| Degrees of freedom | 9 |
| p-value | 0.198 |

**Interpretation.** At a significance level of 0.05, the null hypothesis (train and val distributions are identical) **cannot be rejected** (p=0.198 > 0.05). The splits are statistically balanced, meaning that validation performance will be a reliable proxy for generalization. No stratified re-splitting is necessary.

---

## 5. Bounding Box Characteristics

### 5.1 Size Statistics

| Class | Mean Area (px) | Median Area (px) | Notes |
|---|---|---|---|
| train | 37,708 | 5,768 | Largest mean; high variance (rare, variable-distance objects) |
| car | 9,418 | 1,379 | High mean-to-median ratio indicates right-skewed distribution |
| traffic light | 506 | 263 | Smallest objects overall |
| traffic sign | -- | -- | Comparable to traffic light in scale |

Reference: total image area is 1,280 x 720 = 921,600 pixels.

**Key observation.** The large gap between mean and median area for most classes (e.g., car mean 9,418 vs. median 1,379) indicates heavy right-skew: a small number of close-range objects inflate the mean, but the majority of instances are distant and small. This confirms the small-object finding from the anomaly analysis (Section 8).

### 5.2 Aspect Ratios

| Class | Mean Aspect Ratio (w/h) | Interpretation |
|---|---|---|
| person | 0.46 | Tall and narrow -- expected for upright pedestrians |
| rider | ~0.4-0.5 | Similar to person |
| traffic sign | 1.51 | Wider than tall on average -- includes horizontal regulatory signs |
| car | ~1.5-2.0 | Wider than tall -- standard vehicle profile |

Aspect ratio distributions are class-dependent and should inform anchor box design. A single set of default anchors will be suboptimal; anchor generation via k-means clustering on ground truth boxes (as in YOLOv2+) is recommended.

---

## 6. Annotation Density

| Metric | Value |
|---|---|
| Mean objects per image | 18.44 |
| Median objects per image | 17 |
| Std deviation | 9.62 |
| Minimum | 3 |
| Maximum | 91 |

The distribution is roughly normal with a slight right skew. The most crowded image contains **91 annotated objects**, likely a dense urban scene. High-density images stress NMS (non-maximum suppression) and increase the likelihood of missed detections due to overlapping proposals.

The standard deviation of 9.62 suggests significant variation in scene complexity, which is beneficial for training robust models but requires careful batch construction if uniform difficulty is desired.

---

## 7. Occlusion & Truncation Analysis

### 7.1 Occlusion Rates by Class

| Class | Occlusion Rate |
|---|---|
| rider | 89.1% |
| bike | 84.3% |
| car | 67.7% |
| person | 58.0% |
| traffic light | 3.2% |

**Analysis.** The extremely high occlusion rates for `rider` (89.1%) and `bike` (84.3%) are notable. These two classes are semantically linked -- a rider is typically on a bike, and both are frequently occluded by the vehicle they share the road with or by other vehicles. From a safety perspective, these are vulnerable road users (VRUs), and their high occlusion makes reliable detection especially challenging and especially important.

Traffic infrastructure classes (`traffic light`, `traffic sign`) have low occlusion rates, which is expected given their elevated mounting positions.

### 7.2 Truncation Rates by Class

| Class | Truncation Rate |
|---|---|
| train | 28.5% |
| bus | 18.3% |
| truck | 15.2% |
| traffic light | 2.7% |

Large vehicle classes (`train`, `bus`, `truck`) have the highest truncation rates because they frequently extend beyond the camera's field of view. This is a natural consequence of their physical size relative to the camera frame. Models should be trained to recognize partially visible large vehicles, as truncation is an inherent property of these classes rather than a data quality issue.

---

## 8. Anomalies & Edge Cases

### 8.1 Tiny Bounding Boxes (< 1% of image area)

| Metric | Value |
|---|---|
| Count | 1,286,341 |
| Percentage of all annotations | 87.4% |
| Dominant class | car (662K tiny boxes) |

This is the single most impactful finding for model design. Nearly 9 out of 10 annotations are tiny. This is consistent with dashcam data where most objects are at medium to far range. Models must be evaluated specifically on small-object recall; feature pyramid networks (FPN) or similar multi-scale architectures are essential.

### 8.2 Huge Bounding Boxes (> 80% of image area)

Only **5** annotations exceed 80% of image area. These are likely very close-range vehicles filling the frame. They are too few to affect training but could be useful as hard examples during evaluation.

### 8.3 Extreme Aspect Ratios (> 10:1 or < 1:10)

**474** bounding boxes have extreme aspect ratios. These may represent:
- Partially visible objects at frame edges (consistent with truncation data).
- Annotation artifacts.

At 0.03% of total annotations, their impact on training is negligible, but they should be reviewed during quality assurance.

### 8.4 Summary

| Anomaly Type | Count | % of Total |
|---|---|---|
| Tiny boxes (< 1% area) | 1,286,341 | 87.4% |
| Huge boxes (> 80% area) | 5 | ~0.0% |
| Extreme aspect ratios | 474 | 0.03% |
| Empty images | 0 | 0.0% |

---

## 9. Attribute Patterns

### 9.1 Weather Conditions

**Clear** weather dominates the dataset, followed by overcast, snowy, and rainy conditions. This creates a secondary imbalance axis: models trained predominantly on clear-weather data may degrade under adverse conditions. Weather-specific evaluation is recommended.

### 9.2 Scene Type

**City street** is the dominant scene type, contributing 512K car annotations alone. **Highway** is the second most common. The scene distribution reflects the urban-centric collection of BDD100K. Rural and suburban scenes are under-represented.

### 9.3 Time of Day

**Daytime** dominates, followed by **night**. Dawn and dusk are least represented. Night-time images present unique challenges (headlight glare, reduced contrast) that compound with the small-object problem.

**Cross-attribute note.** The combination of night + rainy + city street represents a worst-case scenario for detection that is likely under-represented in the dataset. Targeted evaluation on this slice is advisable.

---

## 10. Key Findings & Recommendations

### 10.1 Findings Summary

| Finding | Impact | Severity |
|---|---|---|
| 5,400:1 class imbalance (car vs. train) | Naive training will ignore tail classes | High |
| 87.4% of boxes are tiny (< 1% image area) | Standard detectors will miss distant objects | High |
| Rider/bike occlusion > 84% | VRU detection is a safety-critical weak point | High |
| Train/val split is balanced (p=0.198) | Validation metrics are trustworthy | Positive |
| Mean != median box area (heavy right-skew) | Anchor design must account for skewed size distribution | Medium |
| Clear/daytime weather dominates | Model may underperform in adverse conditions | Medium |

### 10.2 Recommendations for Model Training

1. **Loss function.** Use a class-balanced loss (e.g., focal loss, class-balanced cross-entropy with effective number of samples) to counteract the long-tail distribution. Consider per-class loss weighting inversely proportional to frequency.

2. **Architecture.** A multi-scale detection architecture with FPN or similar is non-negotiable given the 87.4% tiny-object rate. High-resolution feature maps must be preserved for small-object detection.

3. **Anchor design.** Generate anchors via k-means clustering on the ground truth bounding boxes rather than using preset defaults. Class-specific aspect ratio priors (e.g., tall/narrow for person, wide for car) will improve proposal quality.

4. **Data augmentation.** Apply scale jitter, mosaic augmentation, and copy-paste augmentation for rare classes (`train`, `motor`, `rider`). Consider oversampling images containing tail-class instances.

5. **Occlusion handling.** Given the high occlusion rates for VRU classes, augmentation strategies that simulate partial occlusion (e.g., random erasing, cutout) may improve robustness.

6. **Evaluation protocol.** Report per-class AP in addition to mAP. Slice evaluation by weather, scene, and time of day. Pay particular attention to `rider`, `bike`, `motor`, and `train` performance, as aggregate metrics will be dominated by the head classes.

7. **Adverse condition testing.** Create explicit evaluation subsets for night + rain and night + snow to quantify degradation under challenging conditions.

---

## 11. Safety-Critical Edge Cases

Beyond statistical anomalies, we identified specific scenarios where an object detection system is most likely to fail with the highest real-world cost. These combine multiple difficulty factors simultaneously.

### 11.1 Tiny Vulnerable Road Users at Night / Rain

Small pedestrians, riders, and cyclists (< 1% image area) in night or rainy conditions. These combine the two hardest detection challenges — small object size and degraded visibility. The dataset contains a significant number of these cases, and they represent the canonical ADAS failure mode: missing a distant pedestrian in poor conditions.

### 11.2 Occluded Pedestrians Near Cars

Pedestrians marked as occluded in images that also contain cars. This simulates people partially hidden behind parked or moving vehicles — a scenario responsible for a significant share of real-world pedestrian collisions. With rider occlusion at 89.1% and person occlusion at 58.0%, this is not an edge case but a dominant condition.

### 11.3 Crowded Night Intersections

City street scenes at night with 30+ annotated objects. High object density combined with low light stresses both the detector and non-maximum suppression, increasing the probability of missed and merged detections.

### 11.4 Truncated Pedestrians at Frame Edge

People partially cut off at the image boundary. These pedestrians are entering or leaving the camera's field of view and are easy for detectors to miss, yet critical for path planning and collision avoidance.

### 11.5 Blurry and Dark Frames with VRUs

Pre-computed image quality metrics (Laplacian variance for blur, mean pixel intensity for brightness) identify frames where image quality itself degrades detection. Blurry frames with pedestrians and underexposed frames containing vulnerable road users are flagged separately in the dashboard.

### 11.6 Rarest Environmental Conditions

The rarest weather + time-of-day combinations (e.g., night + snow, dawn + foggy) have the fewest training samples and represent likely blind spots for any model. These slices should be evaluated independently to quantify performance degradation.

### 11.7 Detection Blind Spots

The 10-class limitation means the following safety-relevant objects are **invisible** to any model trained on this dataset:

- **Animals** (deer, dogs) — a common collision cause in rural and suburban areas
- **Road debris** (fallen cargo, tire fragments)
- **Construction equipment** (cones, barriers, workers outside person class)
- **Emergency vehicles** with non-standard profiles
- **Non-standard VRUs** (wheelchairs, strollers, shopping carts)
- **Fallen pedestrians** — people lying on the road

Any production deployment must augment this detector with additional models or an out-of-distribution anomaly detection system to handle objects outside the training vocabulary.

---

## 12. Notable Samples

The following images are surfaced by the dashboard's analysis tools and serve as representative examples of key dataset characteristics. These can be explored interactively in the Sample Browser and Safety-Critical Edge Cases tabs.

| Category | How to Find | What to Look For |
|---|---|---|
| Most crowded image | Sample Browser → Most Crowded | The densest image contains 91 objects — a packed urban scene stressing NMS |
| Lone `train` instance | Sample Browser → Rare Classes, filter `train` | Only 151 train annotations in the entire dataset; each image is valuable |
| Lone `motor` instance | Sample Browser → Rare Classes, filter `motor` | Motor is the second rarest class at 3,454 annotations |
| Single-class image | Sample Browser → Single-Class Images | Images where every annotation is one class — tests single-class recall |
| Tiny VRU at night | Safety-Critical → Tiny VRU at Night/Rain | Small pedestrian/rider boxes in low-visibility — hardest detection case |
| Occluded pedestrian | Safety-Critical → Occluded Pedestrians Near Cars | Person partially hidden by vehicles — leading ADAS failure scenario |
| Crowded night scene | Safety-Critical → Crowded Night Intersections | 30+ objects at night in city street — NMS stress test |
| Truncated person | Safety-Critical → Truncated Pedestrians | Person entering frame edge — easy to miss, critical to detect |
| Blurry frame | Safety-Critical → Blurry Frames with Pedestrians | Motion blur degrading pedestrian features |
| Dark frame with VRU | Safety-Critical → Dark Frames with VRUs | Underexposed image with cyclists or pedestrians |
| Huge bounding box | Anomalies → Huge Boxes | Very close-range vehicle filling >80% of frame — only 5 in the dataset |
| Extreme aspect ratio | Anomalies → Extreme Aspect Ratios | Heavily truncated or annotation artifacts — 474 cases |

These are not exhaustive but represent the most informative slices for understanding dataset characteristics and anticipating model weaknesses.

---

*Report generated as part of Phase 1: Data Analysis for the Bosch Applied CV coding assignment. All statistics derived from BDD100K train and validation label files (bdd100k_labels_images_train.json, bdd100k_labels_images_val.json). Image quality metrics computed via Laplacian variance and pixel intensity analysis.*
