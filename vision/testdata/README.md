# Test fixtures

- `faces.jpg`: six faces of people who do not exist, from trackid-sim's face pool
  (`tools/facegen`, generated with FLUX.1-schnell, Apache-2.0), placed on a neutral
  background at varied sizes.
- `oracle_faces.json`: what Python's insightface 2.1 `FaceAnalysis` finds in it with the
  AuraFace-v1 weights (SCRFD 10G detector, glintr100 recognizer, `det_size` 640): box,
  score, five landmarks and the raw 512-d embedding per face.
- `oracle_yunet_faces.json`: what OpenCV 5.0.0's `FaceDetectorYN` finds with
  `face_detection_yunet_2026may.onnx` at native resolution, score threshold 0.9, NMS 0.3.

Both were written by `oracle.py`:

```
uv run --python 3.12 --with insightface --with onnxruntime --with opencv-python-headless \
  python oracle.py faces.jpg <root with models/auraface/{scrfd_10g_bnkps,glintr100}.onnx> \
  ../../models/face_detection_yunet_2026may.onnx .
```
