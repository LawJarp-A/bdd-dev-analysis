# Phase 1: Data Analysis

**Dataset:** BDD100K — 79,863 images, 1,472,397 annotations, 10 detection classes, 1280x720.

## Key Statistics

| Metric | Value |
|---|---|
| Train / Val split | 1,286,871 / 185,526 annotations (70K / 10K images) |
| Split balance | Chi-squared p=0.198 — statistically balanced |
| Mean objects per image | 18.4 (min 3, max 91) |
| Empty images | 0 |

## Class Distribution

| Class | Count | % |
|---|---|---|
| car | 815,717 | 55.4% |
| traffic sign | 274,594 | 18.7% |
| traffic light | 213,002 | 14.5% |
| person | 104,611 | 7.1% |
| truck | 34,216 | 2.3% |
| bus | 13,269 | 0.9% |
| bike | 8,217 | 0.6% |
| rider | 5,166 | 0.4% |
| motor | 3,454 | 0.2% |
| train | 151 | 0.01% |

Head-to-tail ratio (car:train) is 5,402:1.

## Main Findings

| Finding | Detail |
|---|---|
| Severe class imbalance | Top 3 classes = 88.5% of data; train has only 151 instances |
| Small object dominance | 87.4% of boxes < 1% of image area |
| High VRU occlusion | rider 89.1%, bike 84.3% occluded |
| Large vehicle truncation | train 28.5%, bus 18.3%, truck 15.2% truncated |
| Weather/time bias | Clear + daytime dominate; night/rain under-represented |

## Safety-Critical Edge Cases

- **Tiny VRUs at night/rain** — small pedestrians and riders in low visibility
- **Occluded pedestrians near cars** — people partially hidden by vehicles
- **Crowded night intersections** — 30+ objects in city streets at night
- **Truncated pedestrians at frame edge** — entering/leaving field of view
- **Blurry/dark frames with VRUs** — image quality degrades detection
- **Rare conditions** — night+snow, dawn+foggy have fewest training samples

## Recommendations

1. Use class-balanced loss to handle 5,402:1 imbalance
2. Multi-scale architecture essential for 87.4% tiny-object rate
3. Evaluate per-class AP, not just aggregate mAP
4. Slice evaluation by weather, scene, and time of day
5. Augmentation for occlusion simulation and rare class oversampling
