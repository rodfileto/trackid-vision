# trackid-vision

In-process face detection, embedding, and video-ingestion for Go, extracted
from [TrackID](https://github.com/rodfileto/trackid). Runs SCRFD detection
and ArcFace-style (AuraFace-v1) embedding directly in the Go process via
[onnxruntime](https://github.com/yalue/onnxruntime_go) — no Python sidecar.

Two packages:

- **`vision`** — the core primitive: decode an image, detect faces, align,
  embed. Re-implements the pre/post-processing insightface's Python
  `FaceAnalysis` does (blob construction, anchor decoding, NMS, 5-point
  similarity alignment) so results match that reference.
- **`videoproc`** — built on top of `vision`: samples frames from a video
  file (via `ffmpeg`/`ffprobe`), detects/embeds faces per frame, clusters
  the embeddings with DBSCAN into one track per person, and picks/crops a
  representative thumbnail per track.

## Runtime prerequisites

Nothing here is bundled or auto-downloaded. On Linux x86_64, CPU-only:

```
scripts/fetch-runtime-deps.sh
```

fetches both pieces below into `models/` and `onnxruntime-gpu/` (both
gitignored, so this runs once per machine/container image) and verifies the
model checksums; re-running it is a no-op once they're present. For another
platform, a CUDA build, or after bumping `onnxruntime_go` in `go.mod`, do it
by hand:

- **ONNX model weights**, under `models/`: an ArcFace-style recognizer
  (AuraFace-v1's `glintr100.onnx`, from
  [`fal/AuraFace-v1`](https://huggingface.co/fal/AuraFace-v1)) and a face
  detector: YuNet (`face_detection_yunet_2026may.onnx`, from the
  [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet))
  or InsightFace's SCRFD (`scrfd_10g_bnkps.onnx`, also mirrored in the fal
  repository). **The files are not under the same terms; see
  [Model licenses](#model-licenses).** Pick the detector with
  `Config.Detector` (`vision.DetectorYuNet` or `vision.DetectorSCRFD`, the
  default for now).
- **`libonnxruntime.so`** (or platform equivalent) — a build of
  [ONNX Runtime](https://github.com/microsoft/onnxruntime), under
  `onnxruntime-gpu/`. `go.mod` pins `onnxruntime_go` to a version that
  targets one specific onnxruntime C API release, not "latest" — check that
  module's own README ("Note on onnxruntime Library Versions") for which
  one, and use the matching release's `lib/libonnxruntime.so.<version>`
  (the real file, not the unversioned symlink next to it) renamed to
  `libonnxruntime.so`. The CPU-only build works fine; a CUDA-capable build
  is required only if you set `UseGPU`.
- **`ffmpeg`/`ffprobe`** on `$PATH` — only needed if you use `videoproc`.

## Usage

```go
import (
    "context"

    "github.com/rodfileto/trackid-vision/vision"
    "github.com/rodfileto/trackid-vision/videoproc"
)

vis, err := vision.NewService(vision.Config{
    Detector:          vision.DetectorYuNet,
    DetectorPath:      "models/face_detection_yunet_2026may.onnx",
    RecognizerPath:    "models/glintr100.onnx",
    SharedLibraryPath: "onnxruntime/libonnxruntime.so", // optional, has a default
    UseGPU:            false,
})
if err != nil {
    log.Fatal(err)
}
defer vis.Close()

// Single image:
faces, err := vis.DetectAndEmbed(imageReader) // []vision.Face, each with an L2-normalized embedding

// Whole video, clustered into per-person tracks:
result, err := videoproc.Process(context.Background(), vis, "clip.mp4", videoproc.Options{
    IntervalSeconds:      1.0,
    EmbedIntervalSeconds: 1.0,
    MinBlur:              50,
    ClusterEPS:           0.4,
    ClusterMinSamples:    3,
}, nil)
```

`vision.Service` is safe to share across goroutines and is meant to be
constructed once (model loading is not cheap) and reused for the process
lifetime.

## Testing

```
go test ./...
```

`videoproc`'s clustering tests are pure math (synthetic embeddings, no
model files needed). `vision`'s test is an oracle regression test against
a real Python insightface run and needs the real model files + onnxruntime
library from "Runtime prerequisites" above (`scripts/fetch-runtime-deps.sh`)
present locally at `../models/*.onnx` and `../onnxruntime-gpu/libonnxruntime.so`
(relative to the `vision/` package dir) — it's skipped implicitly by not
having those files, but will fail loudly if you run it without them.

## License

The code in this repository is MIT — see [LICENSE](LICENSE). The model
weights it loads are **not** covered by that licence and are not bundled.

### Model licenses

| File | Origin | Terms |
|---|---|---|
| `glintr100.onnx` (recognizer) | fal's own AuraFace-v1 weights | Apache-2.0, per the [model card](https://huggingface.co/fal/AuraFace-v1); fal states it was trained on commercially and publicly available data "to enable its usage in commercial setting" (the training data itself is not disclosed). |
| `face_detection_yunet_2026may.onnx` (detector) | OpenCV Zoo's YuNet (Shiqi Yu), the dynamic-shape export of `face_detection_yunet_2023mar.onnx` | MIT for the model and BSD-3-Clause for its [training code](https://github.com/ShiqiYu/libfacedetection.train). It was trained on WIDER Face, whose own licence is CC BY-NC-ND; whether that reaches trained weights is unsettled, and the author states no restriction. |
| `scrfd_10g_bnkps.onnx` (detector) | InsightFace's SCRFD, byte-identical to the one in InsightFace's `antelopev2` pack | **Non-commercial research only.** InsightFace states that its models, and models trained on its annotated data, are for [non-commercial research purposes only](https://github.com/deepinsight/insightface#license). The Apache-2.0 label on the fal repository covers fal's own weights and cannot re-license this file. |

The SCRFD detector is therefore a **non-commercial dependency**. Do not use
it in a commercial product or service without a licence from InsightFace, and
be aware that this affects anything built on this package. Use
`vision.DetectorYuNet` to avoid it (SCRFD is still the default until the
default is flipped). The recognizer's training data is described only as "a
commercial dataset".

### Choosing a detector

Both were run on the same test image. Alignment differs slightly between
them, so the embedding of the same face differs too (cosine 0.88 on average,
0.80 at worst, for ~105 px faces), so do not mix detectors within one
database. This is a smoke test on one image (6 faces), not a benchmark.

SCRFD shrinks any image into a 640x640 canvas; YuNet runs at native
resolution (long side capped at 2048), which matters for small faces in a
large frame. With the test faces pasted into a 1920x1080 frame:

| face size | SCRFD | YuNet, score 0.7 (default) | YuNet, score 0.9 |
|---|---|---|---|
| ~63 px | 6 / 6 | 6 / 6 | 5 / 6 |
| ~37 px | 6 / 6 | 6 / 6 | 4 / 6 |
| ~29 px | 6 / 6 | 6 / 6 | 3 / 6 |
| ~21 px | 0 / 6 | 6 / 6 | 1 / 6 |
| ~16 px | 1 / 6 | 5 / 6 | 0 / 6 |

Neither detector produced a detection on 19 face-free images (wallpapers and
screenshots, an easy negative set). Faces this small carry little identity
information whatever the detector, so check embedding quality, not just
recall.

Verify these terms at the source before relying on them; licences and model
cards change.
