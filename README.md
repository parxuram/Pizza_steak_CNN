<div align="center">

# 🍕 vs 🥩 — Pizza vs. Steak Image Classification

**A hands-on CNN experiment in TensorFlow/Keras exploring binary image classification, data augmentation, and architecture design.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parxuram/pizza-steak-cnn-classification/blob/main/CV_using_CNN.ipynb)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📖 Overview

This project builds and compares multiple deep learning models to classify food images as either **pizza** or **steak**. It's a from-scratch exploration of convolutional neural networks — covering data loading, preprocessing, augmentation, six model training trials (four CNNs, two dense-only baselines), and a custom image upload + prediction pipeline.

The goal wasn't just to hit a number — it was to understand *why* certain design choices (CNN vs. dense layers, augmentation, shuffling) move the needle on a small (1,500 image) dataset.

---

## 📑 Table of Contents

1. [Dataset](#-dataset)
2. [Data Preprocessing & Augmentation](#-data-preprocessing--augmentation)
3. [Models & Results](#-models--results)
4. [Key Takeaways](#-key-takeaways)
5. [Custom Image Prediction](#-custom-image-prediction)
6. [Project Structure](#-project-structure)
7. [Getting Started](#-getting-started)
8. [Dependencies](#-dependencies)
9. [Future Improvements](#-future-improvements)
10. [License](#-license)

---

## 🗂 Dataset

Dataset: [`pizza_steak.zip`](https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip) — a subset of the Food-101 dataset, pre-split into `train`/`test`.

| Split | Pizza | Steak | Total |
|-------|:-----:|:-----:|:-----:|
| Train | 750   | 750   | 1,500 |
| Test  | 250   | 250   | 500   |

```
pizza_steak/
├── train/
│   ├── pizza/   (750 images)
│   └── steak/   (750 images)
└── test/
    ├── pizza/   (250 images)
    └── steak/   (250 images)
```

The notebook downloads and extracts this automatically — no manual setup needed.

---

## 🔄 Data Preprocessing & Augmentation

Images are loaded with `tf.keras.preprocessing.image.ImageDataGenerator`:

- **Rescaling** — pixel values normalized from `[0, 255]` → `[0, 1]`
- **Resizing** — all images resized to `(224, 224)`
- **Batching** — loaded in batches of 32 via `flow_from_directory`

For the augmented training runs, the following transforms were applied to the training set only:

| Parameter | Value |
|---|:---:|
| `rotation_range` | 0.2 |
| `shear_range` | 0.2 |
| `zoom_range` | 0.2 |
| `width_shift_range` | 0.2 |
| `height_shift_range` | 0.2 |
| `horizontal_flip` | `True` |

---

## 🧪 Models & Results

Six models were trained for 5 epochs each (Adam optimizer, `binary_crossentropy` loss). Results below are taken directly from training logs.

| Model | Architecture | Augmentation | Shuffled | Final Train Acc | Final Val Acc | Best Val Acc |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `cnn_model_1` | 4× Conv2D + 2× MaxPool (Tiny VGG style) | ❌ | ✅ | 86.33% | 81.60% | **86.80%** |
| `model_2` | Dense only (Flatten → 4 → 4 → 1) | ❌ | ✅ | 50.00% | 50.00% | 50.00% *(failed to learn)* |
| `model_3` | Dense only (Flatten → 100 → 100 → 100 → 1) | ❌ | ✅ | 79.60% | 72.40% | 75.80% |
| `cnn_model_2` | 3× Conv2D + 3× MaxPool | ❌ | ✅ | 83.87% | **87.00%** | **87.00%** |
| `cnn_model_3` | Same as `cnn_model_2` | ✅ | ❌ | 60.53% | 70.20% | 75.80% |
| `cnn_model_4` ⭐ | Same as `cnn_model_2` | ✅ | ✅ | 79.07% | 86.00% | 86.00% |

> ⭐ **`cnn_model_4`** — trained on shuffled, augmented data — is the model used for final custom-image predictions in the notebook.

### Loss/Accuracy Curves
Training vs. validation loss and accuracy for each CNN trial are plotted inline in the notebook via a reusable `plot_loss_curves()` helper.

---

## 🔑 Key Takeaways

- **Dense-only networks failed outright.** `model_2` never escaped 50% accuracy (equivalent to random guessing) — a small dense network can't learn meaningful spatial features from raw flattened pixels. `model_3`, despite 45M+ parameters, still trailed every CNN.
- **A simple CNN outperformed a much bigger dense network.** `cnn_model_1`/`cnn_model_2` used a fraction of the parameters of `model_3` yet generalized far better — strong evidence for the value of convolutional inductive bias on image data.
- **Augmentation without shuffling hurt performance.** `cnn_model_3` (augmented, unshuffled) underperformed both the non-augmented baseline and the shuffled-augmented version — the model likely saw batches in a biased class order.
- **Shuffling the augmented data recovered performance.** `cnn_model_4` closed most of the gap versus the non-augmented `cnn_model_2`, confirming that data order matters as much as the augmentation itself on small datasets.
- **Best overall validation accuracy: 87.00%** (`cnn_model_2`, no augmentation) — on a dataset this small, a lean architecture without augmentation was competitive with, and briefly ahead of, the augmented pipeline.

---

## 🖼 Custom Image Prediction

The notebook includes an interactive pipeline (built for Google Colab) to test the trained model on your own images:

1. **Upload** an image via `google.colab.files.upload()`
2. **Preprocess** — convert to tensor, resize to `(224, 224)`, normalize to `[0, 1]`, add batch dimension → `(1, 224, 224, 3)`
3. **Predict** using `cnn_model_4`
4. **Display** the image with its predicted label and confidence

**Validation on unseen custom images:**

| Uploaded Image | Predicted Class | Result |
|---|:---:|:---:|
| Pizza photo | `pizza` | ✅ Correct |
| Steak photo | `steak` (99.26% confidence) | ✅ Correct |

```python
img = preprocess_uploaded_image()
predict_and_display_image(img, cnn_model_4, class_names)
```

---

## 📁 Project Structure

```
pizza-steak-cnn-classification/
├── CV_using_CNN.ipynb    # Main notebook — data, training, evaluation, custom predictions
├── README.md              # This file
└── LICENSE
```

---

## 🚀 Getting Started

### Run in Google Colab (recommended)
Click the **Open in Colab** badge at the top of this README — the notebook handles dataset download, extraction, and all dependencies automatically. A GPU runtime (`Runtime → Change runtime type → GPU`) is recommended for faster training.

### Run locally
```bash
git clone https://github.com/parxuram/pizza-steak-cnn-classification.git
cd pizza-steak-cnn-classification
pip install tensorflow numpy matplotlib pillow pandas
jupyter notebook CV_using_CNN.ipynb
```

> Note: the custom image upload cells use `google.colab.files` and will need to be swapped for a local file-picker (e.g. `input()` + `PIL.Image.open()`) if running outside Colab.

---

## 📦 Dependencies

- `tensorflow` (Keras API)
- `numpy`
- `matplotlib`
- `pandas`
- `Pillow` (PIL)
- `google.colab` *(Colab-only, for image upload)*

---

## 🔮 Future Improvements

- [ ] Add transfer learning baseline (e.g. `EfficientNetB0` / `MobileNetV2`) for comparison against from-scratch CNNs
- [ ] Report precision, recall, F1, and a confusion matrix, not just accuracy
- [ ] Train for more epochs with early stopping / learning rate scheduling
- [ ] Replace Colab-specific upload logic with a local/CLI-friendly inference script
- [ ] Export the best model (`cnn_model_4`) as a `.keras` / `SavedModel` artifact for reuse

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built by [Mohit](https://parxuram.github.io) · [GitHub](https://github.com/parxuram)

</div>
