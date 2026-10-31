# Isoline: video voice-over script

Read each part while its slide is on screen. About 5 minutes at a relaxed pace.

## Slide 1: Isoline

Hello, I am Talha Salman (560). This is Isoline, my deep learning semester project. The tagline is the whole idea: it reads the ECG, and it knows when not to.

## Slide 2: The problem

ECGs often travel as printouts and phone photos. Most deep learning models need the clean signal, and image models usually answer with the same confidence whether the photo is crisp or crumpled. A wrong answer that looks sure of itself is the dangerous case. So my question was not only what does this ECG show, but can I trust this picture, and can I trust my own answer.

## Slide 3: The idea: five steps

Isoline works in five steps. First it checks the picture and refuses the bad ones. Then an image model reads the picture directly. A digitizer rebuilds the waveform, and a second, independent network reads that. If the two agree, the answer is marked reliable. Finally a heat map and a one-page report explain the result.

## Slide 4: Dataset: PTB-XL

I used PTB-XL, a public collection of twelve-lead ECGs annotated by cardiologists. I kept the 21,388 records that have a diagnosis, grouped into five classes: normal, heart attack, ST T change, conduction disturbance and hypertrophy. I used the official split, so test patients were never seen in training.

## Slide 5: Making the pictures

PTB-XL has signals, not pictures, so I made the pictures. A renderer prints each record the way a clinic does, with grid, layout and calibration pulses. A camera lab then photographs it badly on purpose, with tilt, shadows, creases, blur, noise and compression, at three severities. That gives 42,776 pictures.

## Slide 6: The pipeline

Here is the full pipeline. The quality gate can stop a bad photo right away and ask for a retake. Otherwise the image model reads it. In parallel, the digitizer rebuilds the waveform, and if it lined up with the grid, the signal model reads that. The cross-check compares the two answers and decides how far to trust the result.

## Slide 7: The two models

The image model is an EfficientNet-B0 with two heads, one for the five diagnoses and one for photo quality. The signal model is a one dimensional residual network with squeeze and excitation blocks, followed by a three layer Transformer. I fine-tuned it on masked signals that show only what a printout actually contains.

## Slide 8: Rebuilding the waveform

The digitizer is classical computer vision. It turns the page into an ink map, traces each lead and converts it to millivolts. To avoid trusting a bad rebuild it computes an alignment score from the calibration pulse. Clean prints score about one. Photos mostly score lower, and below 0.90 the image model answers alone.

## Slide 9: Results on the held-out test set

On the held-out test set the signal model reaches a macro AUC of 0.929, and the image model 0.921, reading only pixels. The quality gate classifies photo quality with 99.1% accuracy. Rebuilding the signal from a picture loses some information, which is why the image model gives the answer and the signal model is the cross-check.

## Slide 10: Answering less, but more reliably

This is the point of the project. If Isoline answers every picture, it is right 82.4% of the time. If it answers only when the photo is good enough and both models agree, it is right 89.6% of the time, on 22.8% of the pictures. The user chooses where to sit on that trade-off.

## Slide 11: Explaining each answer

Every answer comes with a Grad-CAM heat map that shows where the model looked, and a one page report with the verdict, the probabilities, the rebuilt waveforms and plain advice. There are four outcomes: reliable, flagged, image only, and rejected, when the photo is too poor to read.

## Slide 12: Built to be run by anyone

The repository holds the code, the trained weights, the notebook and example pictures, and one command analyses a picture. The website on Vercel shows the dataset, the results and real example outputs, and links to the report, these slides and the video. There is also a live demo on Hugging Face.

## Slide 13: Limits, said plainly

A few honest limits. This is not a medical device. All pictures are rendered from PTB-XL signals, so I have not tested on real clinic photos, and real ones will be harder. Strongly tilted photos cannot yet be cross-checked, and the resolution clips sharp peaks. The next step would be real scans.

## Slide 14: Thank you

Thank you for watching. The code, the weights, the report and the live demo are all linked on the project website.
