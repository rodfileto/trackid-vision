package vision

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// YuNet detector (face_detection_yunet_2026may.onnx, from the OpenCV Zoo;
// model MIT, training code BSD-3-Clause, trained on WIDER Face -- see the
// README's "Model licenses"). The decode below follows OpenCV's
// FaceDetectorYN (modules/objdetect/src/face_detect.cpp).
//
// Unlike SCRFD it takes any input size, so the image is run at its native
// resolution (padded up to a multiple of 32) instead of being shrunk into a
// 640x640 canvas: shrinking is what costs small faces their score.

const (
	// OpenCV's FaceDetectorYN defaults to 0.9. On faces pasted into a 1920x1080
	// frame, 0.7 finds all six test faces down to ~21px (0.9 loses half of
	// them by ~30px) and produced no detections on 19 face-free images.
	yunetDefaultScoreThresh = 0.7
	yunetNMSThresh          = 0.3 // OpenCV's FaceDetectorYN default
	yunetMaxSide            = 2048
	yunetPad                = 32

	yunetInputName = "input"
)

// Output order: cls, obj, bbox, kps -- each at strides 8, 16, 32.
var yunetOutputNames = []string{
	"cls_8", "cls_16", "cls_32",
	"obj_8", "obj_16", "obj_32",
	"bbox_8", "bbox_16", "bbox_32",
	"kps_8", "kps_16", "kps_32",
}

var yunetStrides = [3]int{8, 16, 32}

type yunetDetector struct {
	session     *ort.DynamicAdvancedSession
	scoreThresh float32
}

func newYuNetDetector(path string, opts *ort.SessionOptions) (*yunetDetector, error) {
	session, err := ort.NewDynamicAdvancedSession(path, []string{yunetInputName}, yunetOutputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("load detector model: %w", err)
	}
	return &yunetDetector{session: session, scoreThresh: yunetDefaultScoreThresh}, nil
}

func (d *yunetDetector) Close() { d.session.Destroy() }

func (d *yunetDetector) Detect(img *rgbImage) ([]Detection, error) {
	work, scale := img, float32(1)
	if long := max(img.w, img.h); long > yunetMaxSide {
		scale = float32(yunetMaxSide) / float32(long)
		work = img.resize(int(float32(img.w)*scale+0.5), int(float32(img.h)*scale+0.5))
	}
	padW := (work.w + yunetPad - 1) / yunetPad * yunetPad
	padH := (work.h + yunetPad - 1) / yunetPad * yunetPad

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(padH), int64(padW)), blobBGR(work, padW, padH))
	if err != nil {
		return nil, fmt.Errorf("build detector input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs := make([]ort.Value, len(yunetOutputNames))
	if err := d.session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("run detector: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	var candidates []Detection
	for idx, stride := range yunetStrides {
		var data [4][]float32
		for k := range data {
			t, ok := outputs[idx+3*k].(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("unexpected output type for %s", yunetOutputNames[idx+3*k])
			}
			data[k] = t.GetData()
		}
		cls, obj, bbox, kps := data[0], data[1], data[2], data[3]

		cols, rows := padW/stride, padH/stride
		if len(cls) != rows*cols {
			return nil, fmt.Errorf("stride %d: got %d cells, want %d", stride, len(cls), rows*cols)
		}
		s := float32(stride)
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				i := r*cols + c
				score := float32(math.Sqrt(float64(clamp01f(cls[i]) * clamp01f(obj[i]))))
				if score < d.scoreThresh {
					continue
				}

				b := bbox[i*4 : i*4+4]
				cx := (float32(c) + b[0]) * s
				cy := (float32(r) + b[1]) * s
				w := float32(math.Exp(float64(b[2]))) * s
				h := float32(math.Exp(float64(b[3]))) * s

				k := kps[i*10 : i*10+10]
				var pts [5][2]float32
				for j := range pts {
					pts[j][0] = (k[2*j] + float32(c)) * s / scale
					pts[j][1] = (k[2*j+1] + float32(r)) * s / scale
				}
				candidates = append(candidates, Detection{
					Box:   [4]float32{(cx - w/2) / scale, (cy - h/2) / scale, (cx + w/2) / scale, (cy + h/2) / scale},
					Score: score,
					Kps:   pts,
				})
			}
		}
	}
	return nms(candidates, yunetNMSThresh), nil
}

func clamp01f(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
