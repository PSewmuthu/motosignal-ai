# Research and experimental notebooks

These notebooks are the **first stage of development**. Use them to explore the data, try models, tune parameters, make plots, inspect example frames, and record findings. They are independent experimental documents.

After an experiment produces a useful and repeatable result, implement that idea separately in the application. The production application must not execute or import notebooks. This keeps experimentation easy to change and the application easier to test.

| Notebook                                | Description                                                                                                                                                                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `number-plate-detection-yolo-ocr.ipynb` | Downloaded [Number Plate Detection: YOLO & OCR Kaggle Notebook](https://www.kaggle.com/code/gbiamgaurav/number-plate-detection-yolo-ocr) and edited for fine-tune YOLO for number plate detection and character recognition. |
| `01_dataset_exploration.ipynb`          | Explored video metadata, frame sampling, dataset inspection.                                                                                                                                                                 |
| `02_motorcycle_detection.ipynb`         | Explortion of YOLO motorcycle detection and confidence settings.                                                                                                                                                             |
| `03_motorcycle_tracking.ipynb`          | Did built-in YOLO BoT-SORT and ByteTrack comparison.                                                                                                                                                                         |
| `04_stop_line_analysis.ipynb`           | Experimented stop-line geometry and crossing candidates.                                                                                                                                                                     |
