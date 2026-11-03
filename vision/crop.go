package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
)

// DefaultCropPadding is the extra context most callers want around a
// detected box, as a fraction of its size -- matches what videoproc's face
// thumbnails already use.
const DefaultCropPadding = 0.4

// Crop extracts the region of img covered by box (Detection.Box/Face.Box:
// x1,y1,x2,y2 in img's own pixel space), padded by padding on every side and
// clamped to img's bounds. It returns nil if the padded box is empty or
// falls entirely outside img.
func Crop(img image.Image, box [4]float32, padding float32) image.Image {
	bounds := img.Bounds()
	w, h := box[2]-box[0], box[3]-box[1]
	padX, padY := w*padding, h*padding

	x0 := clampInt(int(box[0]-padX), bounds.Min.X, bounds.Max.X)
	y0 := clampInt(int(box[1]-padY), bounds.Min.Y, bounds.Max.Y)
	x1 := clampInt(int(box[2]+padX), bounds.Min.X, bounds.Max.X)
	y1 := clampInt(int(box[3]+padY), bounds.Min.Y, bounds.Max.Y)
	if x1 <= x0 || y1 <= y0 {
		return nil
	}

	cropped := image.NewRGBA(image.Rect(0, 0, x1-x0, y1-y0))
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			cropped.Set(x-x0, y-y0, color.RGBAModel.Convert(img.At(x, y)))
		}
	}
	return cropped
}

// CropJPEG is Crop followed by JPEG encoding, for callers that just want
// bytes to upload or embed (object storage, a data URI, ...). It returns a
// nil slice, not an error, when Crop finds nothing to encode.
func CropJPEG(img image.Image, box [4]float32, padding float32, quality int) ([]byte, error) {
	cropped := Crop(img, box, padding)
	if cropped == nil {
		return nil, nil
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, cropped, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
