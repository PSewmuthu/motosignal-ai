# Model documentation and integration

## Model artifacts expected

| Component                       | Model                                            | Runtime path variable         | Notes                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Motorcycle detection + tracking | YOLO-family detector with Ultralytics BoT-SORT   | `MOTOSIGNAL_MOTORCYCLE_MODEL` | Uses built-in `model.track(..., persist=True)`; default class index is COCO motorcycle `3`; set `MOTOSIGNAL_MOTORCYCLE_CLASS_ID` to your custom model class index. |
| Rear plate detection            | Fine-tuned YOLO plate detector                   | `MOTOSIGNAL_PLATE_MODEL`      | Runs directly on rear-view CCTV frames. Do not replace this with manual plate crops as the input source.                                                           |
| OCR                             | Pretrained PaddleOCR + optional EasyOCR ensemble | `MOTOSIGNAL_ENABLE_OCR=true`  | Optional; compares conservative image variants and abstains on weak/conflicting reads. Validate against labelled plates from the target jurisdiction.              |

Place weights in `models/` and document filename, SHA-256, training notebook/commit, dataset and license, class map, input size, preprocessing, confidence threshold, evaluation split, precision/recall/mAP, latency, and hardware. Weights are not included in this repository.

## Motorcycle tracking configuration

Install `ultralytics==8.4.203` from `backend/requirements-ml.txt`. This release is available on PyPI as of November 5, 2026 and supports the newer built-in tracker options documented by Ultralytics. The Ultralytics package is published under AGPL-3.0; the repository's MIT license does not replace or override third-party package/model licensing. The default profile is `backend/app/configs/trackers/botsort_standard.yaml` and can be overridden with `MOTOSIGNAL_TRACKER_CONFIG`. A simple ByteTrack comparison profile is also included: `bytetrack_baseline.yaml`.

| Setting                            | Default | Purpose                                                                                         |
| ---------------------------------- | ------: | ----------------------------------------------------------------------------------------------- |
| `MOTOSIGNAL_MOTORCYCLE_CONFIDENCE` |  `0.20` | Starting detector confidence; lower it only if review shows missed motorcycles.                 |
| `MOTOSIGNAL_MOTORCYCLE_NMS_IOU`    |  `0.65` | Detector NMS IoU threshold; tune alongside the model and crowded-scene overlap.                 |
| `MOTOSIGNAL_MOTORCYCLE_IMAGE_SIZE` |   `960` | A practical starting image size; increase if small motorcycles are missed and hardware permits. |
| `MOTOSIGNAL_YOLO_DEVICE`           |    auto | Optional `cpu`, `0`, or `cuda:0`.                                                               |
| `MOTOSIGNAL_FRAME_STRIDE`          |     `1` | Process every frame for accurate trajectories and crossing times.                               |

The standard profile does not enable optional appearance ReID. It is a simple starting point that is easier to understand and run. Compare it with the included ByteTrack profile on the same short video before choosing. No tracker setting is guaranteed to be most accurate for every camera.
The confidence and buffer values are not universal accuracy settings. Evaluate on the actual camera: too-high detector confidence fragments tracks; too-low confidence may introduce false detections; too-long `track_buffer` can reconnect different motorcycles after a long disappearance. `track_buffer` is measured in processed tracker frames, so `frame_stride=1` is recommended for this application.

## Model quality gates

Before claiming the system works end-to-end, verify motorcycle recall on small/distant bikes, ID switches under overlaps, stop-line event precision/recall, plate detection recall under blur/low light/angle/occlusion, OCR exact match and character error rate, false-readable rate, unreadable abstention, and actual CPU/GPU processing FPS. Track detector and OCR confidence thresholds on a held-out validation set.

## Licensing

The code license does not cover model weights or datasets. Check Ultralytics package/model licensing, PaddleOCR/PaddlePaddle/EasyOCR terms, checkpoint provenance, Kaggle notebook source license, and dataset permissions before distribution or commercial use.
