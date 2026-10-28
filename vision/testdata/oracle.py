"""Reference runs for trackid-vision's tests on testdata/faces.jpg: insightface FaceAnalysis with
the AuraFace-v1 weights (SCRFD 10G + glintr100), and OpenCV's FaceDetectorYN (YuNet)."""
import json, sys
import cv2
import insightface
from insightface.app import FaceAnalysis

image, root, yunet, out_dir = sys.argv[1:5]
img = cv2.imread(image)

app = FaceAnalysis(name="auraface", root=root, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))
faces = [{"bbox": [float(v) for v in f.bbox], "det_score": float(f.det_score),
          "kps": [[float(x), float(y)] for x, y in f.kps], "embedding": [float(v) for v in f.embedding]}
         for f in app.get(img)]
json.dump(faces, open(f"{out_dir}/oracle_faces.json", "w"))

h, w = img.shape[:2]
det = cv2.FaceDetectorYN.create(yunet, "", (w, h), 0.9, 0.3, 5000)
_, found = det.detect(img)
yfaces = [{"bbox": [float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3])],
           "kps": [[float(r[4 + 2 * i]), float(r[5 + 2 * i])] for i in range(5)], "score": float(r[14])}
          for r in (found if found is not None else [])]
json.dump(yfaces, open(f"{out_dir}/oracle_yunet_faces.json", "w"))
print("insightface", insightface.__version__, len(faces), "faces; opencv", cv2.__version__, len(yfaces), "faces")
