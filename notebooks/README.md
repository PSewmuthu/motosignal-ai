# Research and experimental notebooks

These notebooks are the **first stage of development**. Use them to explore the data, try models, tune parameters, make plots, inspect example frames, and record findings. They are independent experimental documents.

After an experiment produces a useful and repeatable result, implement that idea separately in the application. The production application must not execute or import notebooks. This keeps experimentation easy to change and the application easier to test.

| Notebook                        | Experiment                                         | Useful visual outputs                                                    |
| ------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| `01_dataset_exploration.ipynb`  | Video metadata, frame sampling, dataset inspection | Sample-frame grids, class counts, image dimensions                       |
| `02_motorcycle_detection.ipynb` | YOLO motorcycle detection and confidence settings  | Bounding-box overlays, confidence plots, missed/false detection examples |
| `03_motorcycle_tracking.ipynb`  | Built-in YOLO BoT-SORT and ByteTrack comparison    | Track overlays, coloured IDs, trajectories, track-duration plots         |
| `04_stop_line_analysis.ipynb`   | Stop-line geometry and crossing candidates         | Stop-line drawn on frames, anchor-point trajectories, crossing timeline  |
