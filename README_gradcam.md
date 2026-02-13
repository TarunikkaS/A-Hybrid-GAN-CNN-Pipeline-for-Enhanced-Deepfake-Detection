# Grad-CAM usage for this project

This repository now includes a simple Grad-CAM tool at `gradcam.py` that can be used to visualize model attention for images.

Quick steps

1. Install dependencies (preferably in a venv):

```bash
python -m pip install -r requirements.txt
```

2. Run Grad-CAM on a single image:

```bash
python gradcam.py --checkpoint best_xception.pth --timm-model xception --image split_dataset/test/real/0001.jpg --output-dir gradcam_out --target-layer features --img-size 299
```

Notes:
- If your checkpoint was trained using a `timm` model (recommended), pass the `--timm-model` name used during training (for example `xception`).
- If you used a custom Xception implementation inside the notebook, you can try `--timm-model xception` as a fallback or update `gradcam.py` to import your model class and construct it instead.
- The default `--target-layer` is `features`. If you know the exact module name in the model (inspect `model.named_modules()`), pass that module name for more accurate maps.

Batch processing:

```bash
python gradcam.py --checkpoint best_xception.pth --timm-model xception --image-dir split_dataset/test/real --output-dir gradcam_out --img-size 299
```

Output:
- Images will be written to the directory provided to `--output-dir`, with filenames `<origname>_gradcam.jpg`.

If you want, I can:
- Update `gradcam.py` to import the exact Xception class from your notebook and auto-load weights from `best_xception.pth` so it works without `--timm-model`.
- Add a small demo notebook cell to `XCEPTION_NET.ipynb` that runs Grad-CAM on a sample from your `split_dataset` and shows results inline.
