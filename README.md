# VggVox
Implementation of [1706.08612] VoxCeleb: a large-scale speaker identification dataset

Speaker Recognition Scores:
Metric  |   Paper   |   Ours
--- |   --- |   ---
Top-5   |   92  |   90.7
Top-1   |   81  |   78

Modifications:
*   Added a Local cepstral mean and variance normalization over a sliding window of 3s

TODO:
*   Add VAD to capture speech frames only