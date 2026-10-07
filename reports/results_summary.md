# Results (test split, 2,203 images)

Joint = fruit and freshness both correct (16 classes, macro precision/recall/F1). Freshness metrics use spoiled as the positive class. Made by notebooks/06_evaluation.ipynb.

## Main comparison

| Model                             |   Accuracy |   Precision |   Recall |   F1-score |   Fruit acc. |   Freshness F1 |   Freshness AUC |   Wrong images |   Params |   ms/img (CPU, batch 1) |
|:----------------------------------|-----------:|------------:|---------:|-----------:|-------------:|---------------:|----------------:|---------------:|---------:|------------------------:|
| Model 1 - Simple CNN (16 classes) |     0.9809 |      0.9804 |   0.9789 |     0.9794 |       0.9909 |         0.9883 |          0.9993 |             42 |  3751632 |                   15.48 |
| Model 2 - Multi-task residual CNN |     0.9796 |      0.9781 |   0.9796 |     0.9785 |       0.9873 |         0.9891 |          0.9993 |             45 |  4986345 |                   20.32 |
| Model 3 - MobileNetV2 fine-tuned  |     0.9973 |      0.9964 |   0.9973 |     0.9968 |       1      |         0.9975 |          0.9998 |              6 |  2505033 |                   35.06 |

## Ablation: data augmentation

| Model                             |   Joint acc. (aug) |   Joint acc. (no aug) |   Joint F1 (aug) |   Joint F1 (no aug) |   Errors (aug) |   Errors (no aug) |
|:----------------------------------|-------------------:|----------------------:|-----------------:|--------------------:|---------------:|------------------:|
| Model 1 - Simple CNN (16 classes) |             0.9809 |                0.9237 |           0.9794 |              0.9199 |             42 |               168 |
| Model 2 - Multi-task residual CNN |             0.9796 |                0.9805 |           0.9785 |              0.98   |             45 |                43 |
| Model 3 - MobileNetV2 fine-tuned  |             0.9973 |                0.9932 |           0.9968 |              0.9927 |              6 |                15 |

## Ablation: fine-tuning depth (Model 3)

| Unfrozen from   |   Unfrozen layers |   Trainable params |   Val joint acc. |   Test joint acc. |   Test joint F1 |   Test freshness F1 |   Test errors |   Train s/step (CPU) |
|:----------------|------------------:|-------------------:|-----------------:|------------------:|----------------:|--------------------:|--------------:|---------------------:|
| none (frozen)   |                 0 |             247049 |           0.9923 |            0.9818 |          0.9797 |              0.9869 |            40 |                0.466 |
| block 16        |                11 |            1126089 |           0.9991 |            0.9936 |          0.9929 |              0.995  |            14 |                0.492 |
| block 13 (main) |                38 |            1910409 |           0.9991 |            0.9973 |          0.9968 |              0.9975 |             6 |                0.531 |
| block 10        |                64 |            2206857 |           0.9991 |            0.9977 |          0.9973 |              0.9979 |             5 |                0.574 |
| block 6         |               100 |            2384841 |           1      |            0.9964 |          0.9954 |              0.9966 |             8 |                0.6   |
| all layers      |               153 |            2436809 |           0.9995 |            0.9986 |          0.9982 |              0.9987 |             3 |                0.931 |
