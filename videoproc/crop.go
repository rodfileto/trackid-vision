package videoproc

import (
	"encoding/base64"
	"image"

	"github.com/rodfileto/trackid-vision/vision"
)

// cropDataURI crops the padded face box out of the frame image and returns
// it as a base64 JPEG data URI, ready for an <img src="..."> in the UI.
func cropDataURI(img image.Image, box [4]float32) (string, error) {
	jpg, err := vision.CropJPEG(img, box, vision.DefaultCropPadding, 90)
	if err != nil {
		return "", err
	}
	if jpg == nil {
		return "", nil
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(jpg), nil
}
