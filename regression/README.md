## 1D Regression

---
### Training
```
python gp.py --mode train --expid cmanp --model cmanp --num_latents_per_layer 128
```
The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
python gp.py --mode eval --expid cmanp --model cmanp --num_latents_per_layer 128
```
Note that `{expid}` must match between training and evaluation since the model will load weights from `results/gp/{model}/{expid}` to evaluate.

## CelebA Image Completion
---

### Data Preparation
Download the CelebA and EMNIST files following previous NP codebases (e.g., [TNP](https://github.com/tung-nd/TNP-pytorch) or [LBANP](https://github.com/BorealisAI/latent-bottlenecked-anp)).

### Training

**CelebA (32 x 32):**
```
python celeba.py --mode train --expid cmanp --model cmanp --resolution 32 --max_num_points 200
```

**CelebA (64 x 64):**
```
python celeba.py --mode train --expid cmanp --model cmanp --resolution 64 --max_num_points 800
```

**CelebA (128 x 128):**
```
python celeba.py --mode train --expid cmanp --model cmanp --resolution 128 --max_num_points 1600
```

### Evaluation

When evaluating for the first time, the evaluation data will be generated and saved in `evalsets/celeba`.

**CelebA (32 x 32):**
```
python celeba.py --mode eval --expid cmanp --model cmanp --resolution 32 --max_num_points 200
```

**CelebA (64 x 64):**
```
python celeba.py --mode eval --expid cmanp --model cmanp --resolution 64 --max_num_points 800
```

**CelebA (128 x 128):**
```
python celeba.py --mode eval --expid cmanp --model cmanp --resolution 128 --max_num_points 1600
```

## EMNIST Image Completion
---

### Training

```
python emnist.py --mode train --expid cmanp --model cmanp
```

### Evaluation

**EMNIST (0-9):**
```
python emnist.py --mode eval --expid cmanp --model cmanp --class_range 0 10
```

**EMNIST (10-46):**

```
python emnist.py --mode eval --expid cmanp --model cmanp --class_range 10 47
```
