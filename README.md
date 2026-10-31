# Isoline

**Reads the ECG. Knows when not to.**

Isoline looks at a photo or scan of a 12-lead ECG and screens it for five heart patterns. Before it says anything, it checks whether the picture is good enough to trust. If the photo is blurry, tilted, shadowed or crumpled, it asks for a better one instead of guessing.

It was built as a semester project for a Deep Learning course, and it is meant for learning, not for medical use.

| | |
|---|---|
| Website | https://isoline-ecg.vercel.app |
| Video walkthrough | https://youtu.be/3K9HgGK2TN4?si=OGC0UvvJCvVO5ap4 |
| Project report | [docs/Isoline_Report.pdf](docs/Isoline_Report.pdf) |
| Slides | [docs/Isoline_Slides.pdf](docs/Isoline_Slides.pdf) |
| LinkedIn post | https://lnkd.in/p/d9brwTtS |

## What it does

1. **Checks the picture.** A small quality gate rates the photo as good, fair or bad. Bad photos stop here.
2. **Reads the image.** An EfficientNet-B0 looks at the whole ECG and scores five patterns: normal (`NORM`), myocardial infarction (`MI`), ST/T-wave change (`STTC`), conduction disturbance (`CD`) and hypertrophy (`HYP`).
3. **Rebuilds the waveform.** A digitizer traces the pen line of each lead and turns it back into millivolt signals.
4. **Cross-checks.** A 1D CNN + Transformer reads the rebuilt signals. When both models agree, the answer is marked *reliable*. When they don't, it is *flagged*.
5. **Explains itself.** A Grad-CAM heatmap shows where the image model looked, and a one-page report pulls everything together.

Every result ends up in one of four outcomes:

| Outcome | What it means |
|---|---|
| **reliable** | The photo is well aligned and both models agree. |
| **flagged** | The two models disagree. Treat the answer as a rough hint. |
| **image-only** | The photo can't be lined up well enough to rebuild the waveform, so only the image model answered. |
| **rejected** | The photo is too poor to read. Isoline asks you to retake it. |

## How well does it work?

Measured on the held-out test split of PTB-XL (patients the models never saw):

| | Result |
|---|---|
| Signal model, full 10 s recordings | macro AUC **0.929** |
| Signal model, only what a printout shows | macro AUC **0.926** |
| Image model, phone photos and clean prints | macro AUC **0.921** |
| Photo-quality gate | **99.1%** accuracy |
| Answering every image | 82.4% top-1 accuracy |
| Answering only when both models agree | **89.6%** top-1 accuracy (22.8% of inputs) |

Plots and the per-class numbers are in [`results/`](results/). "Top-1 accuracy" counts a prediction as right if it matches any of the patient's recorded labels, so it is more forgiving than a strict single-label score.

## Try it

```bash
git clone https://github.com/talhahSalman/isoline-ecg
cd isoline-ecg
pip install -r requirements.txt

python predict.py --image samples/ecg_01.jpg                       # readable summary
python predict.py --image samples/ecg_01.jpg --report report.html  # one-page report
python predict.py --image samples/ecg_01.jpg --json               # machine-readable
python evaluate.py                                                  # run all example pictures
```

It runs on a normal laptop CPU, usually in a couple of seconds per picture. The first run of Grad-CAM is a little slower.

## What's in this repo

```
isoline/            the code
  paper.py            ECG paper renderer and "camera lab" that made the training images
  digitize.py         picture -> waveforms, plus the alignment score
  models.py           the two networks
  pipeline.py         quality check, models, and the final decision
  report.py           the one-page HTML report
predict.py          run Isoline on one picture
evaluate.py         run Isoline on the example pictures
notebooks/          the Colab notebook used to prepare data and train the models
weights/            trained weights and calibration values (committed, no download needed)
results/            metrics and figures from the test set
samples/            example pictures, drawn at random from the test split
space/              the Hugging Face Space (live demo)
site/               the website
docs/               report and slides
scripts/            helpers that package the results and build the website, report and slides, and check the links
```

## How the data was made

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) is a public set of 21,799 clinical 12-lead ECGs, 10 seconds each. We kept the 21,388 records that have a diagnosis and grouped the labels into five classes. The official split was used: folds 1-8 for training, fold 9 for validation, fold 10 for testing.

PTB-XL only contains signals, so we printed them ourselves. Each record became two pictures (42,776 in total): a clean printout on ECG paper, and a simulated phone photo with tilt, perspective, shadows, creases, blur, noise and JPEG compression at three severity levels. The renderer and the degradations are in `isoline/paper.py`. The photo-quality labels come from the severity level that was used.

The signal model was first trained on full 10-second recordings. A printed ECG only shows each lead for 2.5 seconds (lead II for the full 10), so it was then fine-tuned on exactly that layout.

## Training

Everything was trained in Google Colab on a free T4 GPU. The notebook in `notebooks/` runs the whole thing in order: data preparation, the signal model, rendering the images, the image model with its quality head, the digitizer, and the decision rules. The trained weights are in `weights/`, so you do not need to retrain to use Isoline.

## Limits you should know about

- **It is trained on rendered ECGs.** The pictures are PTB-XL signals printed by our own code. It has not been tested on real clinic scans or real phone photos.
- **Tilted phone photos are not solved.** Rebuilding the waveform needs a page that lines up with the template. Most phone photos don't, so they get an image-only answer.
- **Rebuilt signals lose detail.** Our pictures carry less than one pixel per sample, so sharp R-peaks come back slightly clipped. The signal model scores 0.926 AUC on perfect signals but about 0.81 on rebuilt ones. That is why the image model gives the answer and the waveform path acts as a cross-check.
- **Hypertrophy is the hardest class** (AUC about 0.90), and PTB-XL's labels are noisy, which caps what any model can reach.
- **This is not a medical device** and does not give a diagnosis.

## Credits

- Data: [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), Wagner et al., PhysioNet, licensed CC BY 4.0.
- Image model backbone: EfficientNet-B0 from `torchvision`, pretrained on ImageNet.
- Code is released under the MIT license (see `LICENSE`).
