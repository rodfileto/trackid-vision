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

Nothing here is bundled or auto-downloaded — bring your own:

- **ONNX model weights**: an SCRFD detector (e.g. `scrfd_10g_bnkps.onnx`)
  and an ArcFace-style recognizer (e.g. AuraFace-v1's `glintr100.onnx`,
  available from [`fal/AuraFace-v1`](https://huggingface.co/fal/AuraFace-v1)
  on Hugging Face).
- **`libonnxruntime.so`** (or platform equivalent) — a build of
  [ONNX Runtime](https://github.com/microsoft/onnxruntime). The CPU-only
  build works fine; a CUDA-capable build is required if you set `UseGPU`.
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
a real Python insightface run and needs real model files + onnxruntime
present locally at `../models/*.onnx` and `../onnxruntime-gpu/libonnxruntime.so`
(relative to the `vision/` package dir) — it's skipped implicitly by not
having those files, but will fail loudly if you run it without them.

## License

MIT — see [LICENSE](LICENSE).
