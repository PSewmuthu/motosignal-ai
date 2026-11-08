# Research and experimental notebooks

These notebooks are the **first stage of development**. Use them to explore the data, try models, tune parameters, make plots, inspect example frames, and record findings. They are independent experimental documents.

After an experiment produces a useful and repeatable result, implement that idea separately in the application. The production application must not execute or import notebooks. This keeps experimentation easy to change and the application easier to test.

| Notebook                               | Experiment                                                                       | Useful visual outputs                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `01_dataset_exploration.ipynb`         | Video metadata, frame sampling, dataset inspection                               | Sample-frame grids, class counts, image dimensions                       |
| `02_motorcycle_detection.ipynb`        | YOLO motorcycle detection and confidence settings                                | Bounding-box overlays, confidence plots, missed/false detection examples |
| `03_motorcycle_tracking.ipynb`         | Built-in YOLO BoT-SORT and ByteTrack comparison                                  | Track overlays, coloured IDs, trajectories, track-duration plots         |
| `04_stop_line_analysis.ipynb`          | Stop-line geometry and crossing candidates                                       | Stop-line drawn on frames, anchor-point trajectories, crossing timeline  |
| `05_license_plate_detection_ocr.ipynb` | Plate detector and OCR experiment (integrate the Kaggle source with attribution) | Plate boxes, crops, OCR text/confidence examples                         |
| `06_multi_frame_ocr.ipynb`             | Compare OCR predictions across frames                                            | Per-frame prediction table, confidence plot, consensus examples          |
| `07_vehicle_association.ipynb`         | Explore timestamp and vehicle matching                                           | Candidate score charts, time-offset plots, match/no-match examples       |

## Updated OCR experiment workflow

- Notebook 05 samples multiple timestamps from the unchanged source video, runs the existing YOLO plate detector on full frames, saves detected crops, measures crop size/sharpness/brightness/contrast, and visualizes quality-ranked candidates before running pretrained PaddleOCR/EasyOCR.
- Notebook 06 reads `ocr_experiment_outputs/ocr_predictions_by_frame.csv`, keeps distinct-frame observations, visualizes confidence and disagreement, and applies conservative exact-string consensus with abstention.
- These measurements are diagnostic aids, not proof of legibility. OCR confidence is not a calibrated probability, and upscaling cannot recreate missing character detail.

## OCR ground-truth evaluation

Notebook 06 evaluates OCR predictions against crop-specific manual labels in `notebooks/ground_truth/ocr_ground_truth.csv`. It reports exact-match accuracy, character error rate, character accuracy, non-empty prediction rate, false-readable rate, and abstention rate. These labels must be independently verified from the original crop. The current five samples are a diagnostic baseline only.
