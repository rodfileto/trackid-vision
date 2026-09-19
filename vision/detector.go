package vision

import (
	"math"
	"sort"
)

// Detection is one detected face before alignment/embedding.
type Detection struct {
	Box   [4]float32 // x1, y1, x2, y2 in original image pixels
	Score float32
	Kps   [5][2]float32 // 5 landmark points in original image pixels
}

// Detector finds faces in an image and reports, for each, a box, a score and
// the 5 landmarks (both eyes, nose tip, both mouth corners, in that order)
// that alignFace needs to warp the face onto the ArcFace template. Everything
// downstream of Detection is detector-agnostic.
type Detector interface {
	Detect(img *rgbImage) ([]Detection, error)
	Close()
}

// Detect runs the configured detector.
func (s *Service) Detect(img *rgbImage) ([]Detection, error) {
	return s.detector.Detect(img)
}

func scaleKps(kps [5][2]float32, scale float32) [5][2]float32 {
	var out [5][2]float32
	for i := range kps {
		out[i][0] = kps[i][0] / scale
		out[i][1] = kps[i][1] / scale
	}
	return out
}

// letterboxSize replicates insightface's SCRFD.detect() resize math: fit the
// image into targetW x targetH preserving aspect ratio, anchored top-left
// (the rest of the canvas stays zero-padded).
func letterboxSize(srcW, srcH, targetW, targetH int) (newW, newH int, scale float32) {
	imRatio := float64(srcH) / float64(srcW)
	modelRatio := float64(targetH) / float64(targetW)
	if imRatio > modelRatio {
		newH = targetH
		newW = int(float64(newH) / imRatio)
	} else {
		newW = targetW
		newH = int(float64(newW) * imRatio)
	}
	scale = float32(newH) / float32(srcH)
	return
}

func nms(dets []Detection, thresh float32) []Detection {
	sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })

	suppressed := make([]bool, len(dets))
	var keep []Detection
	for i := range dets {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] {
				continue
			}
			if IoU(dets[i].Box, dets[j].Box) > thresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// IoU is the intersection-over-union of two boxes (x1,y1,x2,y2), usable by
// any caller that needs to relate two detections' boxes (e.g. matching a
// cheap detect-only box to a full detect+embed box from a nearby frame).
func IoU(a, b [4]float32) float32 {
	xx1 := max32(a[0], b[0])
	yy1 := max32(a[1], b[1])
	xx2 := min32(a[2], b[2])
	yy2 := min32(a[3], b[3])
	w := max32(0, xx2-xx1+1)
	h := max32(0, yy2-yy1+1)
	inter := w * h
	areaA := (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
	areaB := (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
	return inter / (areaA + areaB - inter)
}

func max32(a, b float32) float32 {
	return float32(math.Max(float64(a), float64(b)))
}

func min32(a, b float32) float32 {
	return float32(math.Min(float64(a), float64(b)))
}
