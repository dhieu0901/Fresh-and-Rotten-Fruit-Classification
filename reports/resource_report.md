# Resource report (CPU)

- CPU only: 20 logical cores, no GPU (TensorFlow 2.21.0 on Windows has no GPU support)
- Keras 3.15.1, batch size 32, input 224x224x3, augmentation on
- Training split: 10320 images (323 steps/epoch); validation: 2203 images
- Step time measured over 20 steps after one warm-up epoch; epoch estimate = training steps + one validation pass
- Inference latency: median over repeated `predict_on_batch` calls on random input

Model 3 rows: fine-tuning cost of each unfreezing depth (stage 2); `frozen` = only the heads are trained.

| Model | Total params | Trainable | Non-trainable | s / step | est. min / epoch | ms / image (batch 1) | ms / image (batch 32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| model1_simple_cnn | 3,751,632 | 3,751,632 | 0 | 0.691 | 3.9 | 15.48 | 6.01 |
| model2_multitask_cnn | 4,986,345 | 4,980,521 | 5,824 | 0.877 | 5.0 | 20.32 | 6.35 |
| model3_mobilenet_v2_frozen | 2,505,033 | 247,049 | 2,257,984 | 0.466 | 3.0 | 34.66 | 12.54 |
| model3_ft_block16 | 2,505,033 | 1,126,089 | 1,378,944 | 0.492 | 3.1 | 34.83 | 12.5 |
| model3_mobilenet_v2_finetune | 2,505,033 | 1,910,409 | 594,624 | 0.531 | 3.3 | 35.06 | 12.51 |
| model3_ft_block10 | 2,505,033 | 2,206,857 | 298,176 | 0.574 | 3.6 | 36.13 | 12.97 |
| model3_ft_block6 | 2,505,033 | 2,384,841 | 120,192 | 0.6 | 3.7 | 34.65 | 12.62 |
| model3_ft_all | 2,505,033 | 2,436,809 | 68,224 | 0.931 | 5.5 | 36.03 | 12.72 |
