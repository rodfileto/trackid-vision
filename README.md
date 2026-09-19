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

- **ONNX model weights**: an SCRFD detector (`scrfd_10g_bnkps.onnx`) and an
  ArcFace-style recognizer (AuraFace-v1's `glintr100.onnx`), both from
  [`fal/AuraFace-v1`](https://huggingface.co/fal/AuraFace-v1) on Hugging
  Face (ungated) — save them under `models/`. **The two files are not under
  the same terms; see [Model licenses](#model-licenses).**
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
    DetectorPath:      "models/scrfd_10g_bnkps.onnx",
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
| `scrfd_10g_bnkps.onnx` (detector) | InsightFace's SCRFD, byte-identical to the one in InsightFace's `antelopev2` pack | **Non-commercial research only.** InsightFace states that its models, and models trained on its annotated data, are for [non-commercial research purposes only](https://github.com/deepinsight/insightface#license). The Apache-2.0 label on the fal repository covers fal's own weights and cannot re-license this file. |

The SCRFD detector is therefore a **non-commercial dependency**. Do not use
it in a commercial product or service without a licence from InsightFace, and
be aware that this affects anything built on this package. `DetectorPath` is
configurable: to avoid the restriction, supply a detector whose licence
permits your use (it must provide the 5 facial landmarks the alignment step
needs). Replacing the default detector is tracked as follow-up work.

Verify these terms at the source before relying on them; licences and model
cards change.
