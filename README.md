# Fresh and Rotten Fruit Classification

Course project for Deep Learning at the National Economics University (NEU), instructor Dr. Nguyễn Thị Kim Ngân.

Group: Triệu Hải Đăng Trình, Trương Hoàng Tùng, Nguyễn Đại Quân, Nguyễn Dương Hiếu.

We classify photos of 8 kinds of fruit as fresh or spoiled and compare three CNNs built with TensorFlow/Keras:

1. **Model 1** - a simple Sequential CNN (5 conv blocks + dense layers) with one softmax over the 16 classes (8 fruits x fresh/spoiled).
2. **Model 2** - a multi-task CNN made of residual blocks (Functional API) with two outputs: fruit type (softmax) and freshness (sigmoid).
3. **Model 3** - MobileNetV2 pretrained on ImageNet with the same two outputs, trained in two stages: first only the new layers, then fine-tuning from block 13.

## Data

[Spoiled and Fresh Fruit Inspection Dataset](https://data.mendeley.com/datasets/6ps7gtp2wg/1) (Pachon Suescun et al., 2020, CC BY 4.0): 16,000 RGB images of 224 x 224, 1,000 per class. Fruits: banana, lemon, lulo, mango, orange, strawberry, tamarillo, tomato.

- 1,273 images are byte-identical copies and one photo is stored under both labels. After removing them 14,726 images are left.
- The photos were taken in bursts, so neighbouring images are often almost the same. With a random split by image, 6.5 % of the test images would have a near-copy in the training set. We group related images (perceptual hash + small thumbnails) and put whole groups into one split, stratified by class: 10,320 train / 2,203 validation / 2,203 test images. The split used for all experiments is in `data/splits/`.

## Results (test set, 2,203 images)

"Joint" means that both the fruit and the freshness are right (16 classes).

| Model | Joint accuracy | Joint F1 (macro) | Fruit accuracy | Freshness F1 | Wrong images | Parameters |
|---|---|---|---|---|---|---|
| Model 1 - simple CNN | 98.09 % | 0.979 | 99.09 % | 0.988 | 42 | 3.75 M |
| Model 2 - multi-task residual CNN | 97.96 % | 0.979 | 98.73 % | 0.989 | 45 | 4.99 M |
| Model 3 - MobileNetV2 fine-tuned | **99.73 %** | **0.997** | **100 %** | **0.998** | **6** | 2.51 M |

![Test error rates](figures/model_comparison.png)

Ablations:

- **Data augmentation** (flips, rotation, zoom, brightness, contrast). Without it Model 1 drops to 92.37 % (168 wrong images): it overfits, since most of its weights are in the dense layer after Flatten. Model 2, which uses global average pooling, stays the same (98.05 %), and Model 3 gets a little worse (99.32 %).
- **Number of fine-tuned layers** (Model 3, all runs start from the same stage-1 weights): only the new layers 98.18 %, from block 16 99.36 %, block 13 99.73 %, block 10 99.77 %, block 6 99.64 %, all layers 99.86 %. Fine-tuning helps a lot for freshness. After block 13 the differences are only a few images, while training all layers is almost twice as slow.

Grad-CAM shows that Model 3 looks at the fruit itself, while the two smaller CNNs sometimes look at the background:

![Grad-CAM of the freshness decision](figures/gradcam/gradcam_spoiled.png)

More tables are in `reports/` (`results_summary.md`, `error_analysis.md`, `final_metrics.csv`) and in the notebooks.

## Repository

```text
notebooks/
  01_data_preparation.ipynb        download, cleaning, duplicates, grouped split, leakage check
  02_input_pipeline.ipynb          tf.data batches, labels, augmentation
  03_model1_simple_cnn.ipynb       Model 1 + ablation without augmentation
  04_model2_multitask_cnn.ipynb    Model 2 + ablation without augmentation
  05_model3_mobilenet_v2.ipynb     Model 3 (two stages) + fine-tuning depth and augmentation ablations
  06_evaluation.ipynb              test metrics, confusion matrices, model size and speed
  07_error_analysis_gradcam.ipynb  errors of the three models, Grad-CAM
  08_demo.ipynb                    predictions for new images
src/           code shared by several notebooks (paths and labels, tf.data pipeline, dataset audit, metrics, Grad-CAM)
data/splits/   the train/val/test split used for all experiments
demo/          a few test images for the demo
figures/, reports/   results written by the notebooks
```

## How to run

Python 3.12 with TensorFlow 2.21 (everything was trained on a CPU).

```bash
pip install -r requirements.txt
```

Then run the notebooks in order (VS Code or Jupyter). Notebook 01 downloads the dataset (about 95 MB) and makes the same split as in `data/splits/`. The model notebooks 03-05 have a switch `TRAIN` at the top: with `TRAIN = True` they train the models from scratch (several hours each on a CPU), with `TRAIN = False` they load the saved models from `models/`. The trained weights are not in the repository because of their size, so the notebooks are saved with their outputs.
