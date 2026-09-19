package vision

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// SCRFD detector (scrfd_10g_bnkps.onnx, from InsightFace). NOTE: InsightFace
// releases its models for non-commercial research use only; see the README's
// "Model licenses".

const (
	detInputSize = 640
	detMean      = 127.5
	detStd       = 128.0
	detThresh    = 0.5
	nmsThresh    = 0.4

	detectorInputName = "input.1"
)

// Output order matches the scrfd_10g_bnkps.onnx graph exactly: 3 strides x
// (score, bbox, kps), see internal/vision's onnx introspection notes.
var detectorOutputNames = []string{
	"448", "471", "494", // scores, strides 8/16/32
	"451", "474", "497", // bbox distances, strides 8/16/32
	"454", "477", "500", // kps distances, strides 8/16/32
}

var featStrides = [3]int{8, 16, 32}

const numAnchors = 2

type scrfdDetector struct {
	session *ort.DynamicAdvancedSession
}

func newSCRFDDetector(path string, opts *ort.SessionOptions) (*scrfdDetector, error) {
	session, err := ort.NewDynamicAdvancedSession(path, []string{detectorInputName}, detectorOutputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("load detector model: %w", err)
	}
	return &scrfdDetector{session: session}, nil
}

func (d *scrfdDetector) Close() { d.session.Destroy() }

func (d *scrfdDetector) Detect(img *rgbImage) ([]Detection, error) {
	newW, newH, detScale := letterboxSize(img.w, img.h, detInputSize, detInputSize)
	resized := img.resize(newW, newH)

	canvas := &rgbImage{w: detInputSize, h: detInputSize, pix: make([]uint8, detInputSize*detInputSize*3)}
	for y := 0; y < newH; y++ {
		copy(canvas.pix[(y*detInputSize)*3:(y*detInputSize+newW)*3], resized.pix[(y*newW)*3:(y*newW+newW)*3])
	}

	blob := blobNCHW(canvas, detInputSize, detInputSize, detMean, detStd)
	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, detInputSize, detInputSize), blob)
	if err != nil {
		return nil, fmt.Errorf("build detector input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs := make([]ort.Value, len(detectorOutputNames))
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
	for idx, stride := range featStrides {
		scoreTensor, ok := outputs[idx].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected output type for scores at stride %d", stride)
		}
		bboxTensor, ok := outputs[idx+3].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected output type for bbox at stride %d", stride)
		}
		kpsTensor, ok := outputs[idx+6].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected output type for kps at stride %d", stride)
		}

		scores := scoreTensor.GetData()
		bboxes := bboxTensor.GetData()
		kpsData := kpsTensor.GetData()

		height := detInputSize / stride
		width := detInputSize / stride

		for cell := 0; cell < height*width; cell++ {
			cy := float32((cell / width) * stride)
			cx := float32((cell % width) * stride)

			for a := 0; a < numAnchors; a++ {
				row := cell*numAnchors + a
				score := scores[row]
				if score < detThresh {
					continue
				}

				bb := bboxes[row*4 : row*4+4]
				x1 := cx - bb[0]*float32(stride)
				y1 := cy - bb[1]*float32(stride)
				x2 := cx + bb[2]*float32(stride)
				y2 := cy + bb[3]*float32(stride)

				kp := kpsData[row*10 : row*10+10]
				var kps [5][2]float32
				for j := 0; j < 5; j++ {
					kps[j][0] = cx + kp[2*j]*float32(stride)
					kps[j][1] = cy + kp[2*j+1]*float32(stride)
				}

				candidates = append(candidates, Detection{
					Box:   [4]float32{x1 / detScale, y1 / detScale, x2 / detScale, y2 / detScale},
					Score: score,
					Kps:   scaleKps(kps, detScale),
				})
			}
		}
	}

	return nms(candidates, nmsThresh), nil
}
