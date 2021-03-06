**Lightweight OpenPose implemented in TensorFlow 2.0**

This repository provides a clean implementation of [Lightweight OpenpPose](https://arxiv.org/pdf/1811.12004.pdf)
in TensorFlow 2. I try to implement every possible thing in OOP. Hope it helps.
You could find the original implementation in PyTorch [here](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

***Key Features***

- [x] Dataset Loader of COCO2017 using `tf.data` input pipeline
- [x] Custom training loop using `tf.GradientTape`
- [x] Custom model implementation using `tf.keras`
- [x] Evaluation code which uses COCO format
- [x] Various exporting options to SavedModel, FrozenGraph, TFLite, MNN
- [x] Inference code for each exported format
- [x] `tf.function` and `Concrete Function`
- [x] TensorBoard integration

***Usage***
****Data Preparation****
```bash
python utils/prepare_annotations.py --labels .../person_keypoints_train2017.json 
--output-name train.pkl
```
****Training****

```bash
python script/train.py configs/train.yaml
```

****Evaluating****

```bash
python script/val.py configs/val.yaml
```

****Export****

```bash
python utils/export.py configs/export.yaml
```

****Inference****
```bash
python serving/native configs/val.yaml
```

***Note:***
If you have any question, feel free to open an issue or reach me out at this [email](minhhoangbui.vn@gmail.com)