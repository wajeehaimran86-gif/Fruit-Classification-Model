# Fruit Classification Model — Focused 3-Fruit Version

This version keeps the original custom One-vs-Rest Linear SVM, Softmax confidence, handcrafted 51-D features, and the multi-fruit OpenCV detector. The assignment is focused on **Apple, Banana, and Orange** as requested.

## Robust pipeline

Training: foreground isolation → fruit crop → resize → rotation/scale/lighting/blur/hue augmentation → five synthetic backgrounds → original 51-D features → standardization → custom OvR SVM.

Single-image inference: the same foreground/crop/features are used, then a compact real-world appearance prior (color + local saturation + edges + compactness) is fused with the SVM decision scores before Softmax. This addresses the white-background Fruits-360 → real-photo domain gap without replacing the original SVM architecture.

## Run

```powershell
python train.py
python evaluate.py "C:\path\to\fruit.jpg"
python detect_multi.py "C:\path\to\multi_fruit.jpg"
```

`evaluate.py` is the main single-image prediction entry point. `detect_multi.py` remains the bonus multi-fruit pipeline.

## Streamlit UI

After installing the requirements, run:

```bash
streamlit run app.py
```

Then upload a single fruit image. The UI shows the uploaded image, the
fruit-focused preprocessing preview, the final SVM + Softmax prediction, and
all class probabilities. Enable **Run multi-fruit detection** to use the
retained multi-fruit detector.

The current focused model classes are **Apple, Banana, and Orange**.
