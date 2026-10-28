module github.com/rodfileto/trackid-vision

go 1.27.1

require (
	github.com/yalue/onnxruntime_go v1.24.0
	golang.org/x/image v0.46.0
)

// v0.1.0-v0.4.0 carried a test image that was not free to publish (a copyrighted photo of real
// people); their history was rewritten without it. Use v0.5.0 or later.
retract [v0.1.0, v0.4.0]
