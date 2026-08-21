# Seva Tech

A crop leaf disease scanner that runs a real convolutional neural network
**entirely client-side in the browser**, via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
No backend, no API calls — the model weights ship with the page and inference
happens on-device.

## Project layout

```
.
├── index.html    # Markup only — links to styles.css and script.js
├── styles.css    # All design system CSS (colors, layout, animations)
├── script.js     # App logic + the trained CNN weights, base64-embedded
└── README.md     # This file
```

## Running it

`script.js` is loaded via a relative `<script src="script.js">` tag, and the
page also loads a Google Fonts stylesheet and the `onnxruntime-web` library
from a CDN. Browsers block relative-path script/style loading under the
`file://` protocol (CORS), so **you need to serve the folder over HTTP** —
you can't just double-click `index.html`.

Easiest option, from inside this folder:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Any other static file server works too (`npx serve`, VS Code's Live Server
extension, nginx, etc). An internet connection is still needed on first load
to fetch the Google Fonts CSS and the `onnxruntime-web` runtime from
`cdn.jsdelivr.net` — after that, the model itself runs fully offline since
its weights are embedded in `script.js`.

## What's real here

- The CNN is a genuine 5-layer VGG-style architecture (~2.9M parameters),
  trained from scratch in PyTorch and exported to ONNX.
- `script.js` decodes the embedded base64 weights into an ONNX Runtime
  session at page load, then runs a real forward pass (resize → ImageNet
  normalization → conv/pool stack → softmax) on whatever photo you upload.
- Confidence scores and the "other possibilities" list are the model's
  actual softmax output — not scripted values.

## What's scaled down (read this before demoing it to anyone)

The checkpoint embedded in `script.js` is a **small demo checkpoint**, not a
production model:

| | This checkpoint | Full run |
|---|---|---|
| Classes | 10 | 38 (all of PlantVillage) |
| Images per class | 100 | ~1,400 avg |
| Epochs | 10 | 25+ |
| Hardware | 1 CPU core | GPU (e.g. T4) |
| Training time | ~2.5 minutes | 30–60 minutes |
| Validation accuracy | ~55% | ~90–97% (typical for this architecture on PlantVillage) |

It was trained just enough to prove the pipeline genuinely learns from real
images — it will only ever predict one of its 10 known classes, and its
per-image confidence should be read as "best guess among 10 options," not a
clinical diagnosis.

The 10 classes it currently recognizes:

- Apple — Black Rot
- Cherry — Healthy
- Corn — Gray Leaf Spot (Cercospora)
- Corn — Common Rust
- Grape — Healthy
- Orange — Citrus Greening (HLB)
- Peach — Healthy
- Potato — Late Blight
- Raspberry — Healthy
- Tomato — Bacterial Spot

## Training the full 38-class model

The original training pipeline (`src/train.py`, `src/model.py`,
`src/dataset.py` from the `agriscan_cnn` project this site is built on)
supports the complete PlantVillage dataset out of the box. To train it for
real:

```bash
git clone --filter=blob:none --sparse --depth 1 \
  https://github.com/spMohanty/PlantVillage-Dataset.git
cd PlantVillage-Dataset
git sparse-checkout set raw/color

pip install -r requirements.txt
python src/train.py \
  --data-root /path/to/PlantVillage-Dataset/raw/color \
  --epochs 25 \
  --batch-size 64 \
  --img-size 128
```

On a single free-tier Colab T4 GPU this takes roughly 30–60 minutes and
should land around 90%+ validation accuracy.

## Swapping in a new checkpoint

Once you have a better checkpoint:

1. Export it: `python src/export_onnx.py --checkpoint checkpoints/best_model.pt --out model.onnx`
2. Merge the external weights file into one self-contained file (avoids
   fetch/path issues in the browser):
   ```python
   import onnxruntime as ort
   so = ort.SessionOptions()
   so.optimized_model_filepath = "model.ort"
   so.add_session_config_entry("session.save_model_format", "ORT")
   so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
   ort.InferenceSession("model.onnx", sess_options=so, providers=["CPUExecutionProvider"])
   ```
3. Re-embed it in `script.js`: replace the `MODEL_ORT_BASE64` constant with
   `base64.b64encode(open("model.ort","rb").read()).decode()`.
4. Update `CLASS_NAMES` and `DISEASE_INFO` in `script.js` (and the library
   section in `index.html`, if you want) to match the new class list —
   order matters, since it must match the model's output index order exactly.

If the new checkpoint gets too large to comfortably embed as base64 (say,
much past ~15MB), it's worth switching to fetching `model.ort` as a separate
file instead of inlining it — same `ort.InferenceSession.create()` call,
just passed a URL instead of a `Uint8Array`.

## Disclaimer

This tool is a demo. It is not a substitute for a trained agricultural
professional — for a confirmed diagnosis, consult your local agricultural
extension office.

