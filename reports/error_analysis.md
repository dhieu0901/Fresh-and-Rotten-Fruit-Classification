# Error analysis (test split, 2,203 images)

Joint error = fruit type or freshness (or both) wrong. Freshness: spoiled is the positive class, so a
false alarm is a fresh fruit predicted spoiled and a miss is a spoiled fruit predicted fresh.
Made by notebooks/07_error_analysis_gradcam.ipynb; images of the errors: figures/misclassified_samples.png,
Grad-CAM of the freshness errors: figures/gradcam/gradcam_errors.png.

## Error types

| Model | Joint errors | Fruit only | Freshness only | Both | False alarms (fresh -> spoiled) | Misses (spoiled -> fresh) |
|---|---|---|---|---|---|---|
| Model 1 | 42 | 14 | 22 | 6 | 17 | 11 |
| Model 2 | 45 | 19 | 17 | 9 | 17 | 9 |
| Model 3 | 6 | 0 | 6 | 0 | 2 | 4 |

## Joint errors per fruit

| Fruit | Test images | Model 1 | Model 2 | Model 3 |
|---|---|---|---|---|
| Banana | 300 | 0 | 0 | 0 |
| Lemon | 232 | 6 | 0 | 0 |
| Lulo | 232 | 6 | 2 | 2 |
| Mango | 300 | 6 | 8 | 0 |
| Orange | 300 | 0 | 0 | 1 |
| Strawberry | 300 | 1 | 2 | 0 |
| Tamarillo | 248 | 9 | 11 | 3 |
| Tomato | 291 | 14 | 22 | 0 |

## Most frequent confusions

| Model | True class | Predicted | Images |
|---|---|---|---|
| Model 1 | Lulo fresh | Lulo spoiled | 5 |
| Model 1 | Mango spoiled | Strawberry spoiled | 5 |
| Model 1 | Tomato fresh | Tomato spoiled | 4 |
| Model 1 | Tomato spoiled | Lulo spoiled | 4 |
| Model 1 | Lemon fresh | Lemon spoiled | 3 |
| Model 2 | Tomato fresh | Tamarillo spoiled | 7 |
| Model 2 | Tomato fresh | Tamarillo fresh | 6 |
| Model 2 | Tamarillo fresh | Tamarillo spoiled | 6 |
| Model 2 | Tamarillo spoiled | Tamarillo fresh | 4 |
| Model 2 | Mango fresh | Tamarillo fresh | 4 |
| Model 3 | Lulo spoiled | Lulo fresh | 2 |
| Model 3 | Tamarillo spoiled | Tamarillo fresh | 2 |
| Model 3 | Orange fresh | Orange spoiled | 1 |
| Model 3 | Tamarillo fresh | Tamarillo spoiled | 1 |

## Errors shared between the models

| Misclassified by | Images |
|---|---|
| exactly 1 model | 61 |
| exactly 2 models | 16 |
| exactly 3 models | 0 |
| both Model 1 and Model 2 | 14 |
| both Model 1 and Model 3 | 0 |
| both Model 2 and Model 3 | 2 |

## Confidence of the freshness errors

| Model | Freshness errors | Uncertain (P(spoiled) 0.2-0.8) | Confidently wrong | Median P(spoiled), false alarms | Median P(spoiled), misses |
|---|---|---|---|---|---|
| Model 1 | 28 | 15 | 13 | 0.79 | 0.29 |
| Model 2 | 26 | 12 | 14 | 0.88 | 0.19 |
| Model 3 | 6 | 1 | 5 | 0.71 | 0.08 |

## Every test error of Model 3

| Image | True class | Predicted | P(spoiled) | P(true fruit) |
|---|---|---|---|---|
| F_Orange/642.jpg | Orange fresh | Orange spoiled | 0.81 | 1.0 |
| F_Tamarillo/619.jpg | Tamarillo fresh | Tamarillo spoiled | 0.61 | 1.0 |
| S_Lulo/28.jpg | Lulo spoiled | Lulo fresh | 0.07 | 1.0 |
| S_Lulo/345.jpg | Lulo spoiled | Lulo fresh | 0.0 | 0.8 |
| S_Tamarillo/762.jpg | Tamarillo spoiled | Tamarillo fresh | 0.1 | 1.0 |
| S_Tamarillo/763.jpg | Tamarillo spoiled | Tamarillo fresh | 0.09 | 1.0 |
