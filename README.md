

# WorldLevel_DIP – Word-Level ASL Recognition (Demo)

This project is a **tiny prototype** for word-level American Sign Language (ASL) recognition.

Pipeline idea:

1. **Video → Hand keypoints** with MediaPipe Hands.
2. **Keypoint sequence → Feature tensor** with fixed length.
3. **Sequence model (GRU)** in PyTorch to classify the sign (word).

Currently only use a small subset of WLASL (e.g. `good, happy, you, please, thank you`) to keep things simple.

---

## How it works (high level)

### 1. Data

- Raw videos are stored in:

  ```text
  data_raw/<gloss_name>/*.mp4

* A small CSV (`meta/filelist.csv`) maps each video to its label:

  ```text
  video_path,label_name
  data_raw/good/xxx.mp4,good
  ...
  ```

### 2. Preprocessing

Script: `src/preprocess.py`

For each video:

1. Read all frames with OpenCV.
2. Run **MediaPipe Hands** on each frame.
3. For each detected hand, take 21 landmarks × (x, y, z) → **63-dim vector**.
4. Build a sequence of these vectors.
5. Resize / pad the sequence to a fixed length `SEQ_LEN` (e.g. 32 frames).
6. Save to `data_npy/<label_name>/*.npy` as a tensor of shape `(SEQ_LEN, 63)`.

So every video becomes a small “time series” of hand keypoints.

### 3. Dataset & Dataloaders

Script: `src/dataset.py`

* Reads all `.npy` files from `data_npy/`.
* Creates `(sequence, label)` pairs.
* Splits them into **train / val / test**.
* Returns PyTorch `DataLoader`s.

### 4. Model

Script: `src/model.py`

* A simple **GRU classifier**:

  * Input: sequence `(T, 63)`
  * GRU → final hidden state
  * Fully-connected layer → logits over `num_classes`

So the model learns how a sequence of hand positions corresponds to a word.

### 5. Training

Notebook: `notebooks/03_train_word_level.ipynb`

* Define model, loss (cross-entropy), optimizer (Adam).
* Use helper functions in `src/train_utils.py`:

  * `train_one_epoch(...)`
  * `eval_one_epoch(...)`
* Log train/val loss and accuracy per epoch.
* Optionally save best model as `word_level_gru.pth`.

---

## Typical run order

1. **Prepare folders** (only once)

   ```bash
   python src/setup_folders.py
   ```

2. **Prepare subset from WLASL** (optional helper)

   ```bash
   python src/prepare_wlasl_subset.py
   ```

3. **Build `filelist.csv`**

   ```bash
   python src/make_filelist.py
   ```

4. **Preprocess videos → keypoints**

   ```bash
   python src/preprocess.py
   ```

5. **Train model** (in notebook)

   * Open `notebooks/03_train_word_level.ipynb`
   * Run all cells.

---

## Notes

* This is a **learning / demo** project, not a production model.
* Accuracy is limited by:

  * very small number of videos,
  * only hand keypoints (no face/pose),
  * simple GRU architecture.
* To improve: add more classes, more samples, more features, and a stronger model.
