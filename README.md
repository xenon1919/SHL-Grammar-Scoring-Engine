# SHL Grammar Scoring Engine

A machine learning pipeline that predicts a **grammar score from 0 to 5** for short spoken audio clips (45 to 60 seconds each). Built for the SHL Hiring Assessment 2026 competition on Kaggle.

The pipeline transcribes each clip with Whisper, measures the transcript and the raw audio in several ways, and blends a small set of regression models. Everything lives in one documented notebook, including the evaluation, plots and a written report.

## Task

* **Input:** an audio file (.wav), 45 to 60 seconds of speech.
* **Output:** a continuous grammar score between 0 and 5, following a MOS Likert rubric (1 means very limited control of grammar, 5 means high accuracy and complex structures).
* **Data:** 769 labelled training clips and 216 test clips.
* **Metrics:** RMSE and Pearson correlation.

## Results

Final run of the notebook, with 5 fold stratified cross validation:

* **Training RMSE:** 0.2611 (Pearson 0.9831)
* **Cross validated RMSE:** 0.5728
* **Cross validated Pearson:** 0.8932
* **Public leaderboard (earlier version, before the DeBERTa model was added):** 0.4376, ranked 14th of about 31 on the first submission

The cross validated numbers are the honest estimate of performance. The training RMSE is lower because the models have seen those clips, and it is reported because the competition requires it.

## Approach

1. **Transcription.** Whisper (small, via faster_whisper) produces the transcript, word timestamps and confidence values.
2. **Handcrafted features.** Transcript length, sentence statistics, vocabulary richness, filler words, repetition, connectives, plus speaking rate and pause statistics from the timestamps.
3. **Language model fluency.** GPT 2 token level perplexity statistics, used as a grammaticality signal.
4. **Grammar rule errors.** LanguageTool error rates per word. This step needs Java and internet, and it falls back to zeros if they are unavailable.
5. **Embeddings.** A sentence transformer on the transcript, and wav2vec2 on the raw audio (mean and standard deviation pooling from a middle and the last layer).
6. **Transformer fine tuning.** DeBERTa v3 base fine tuned on the transcripts with a regression head, trained inside the same five folds as the other models.
7. **Modelling.** Ridge, SVR and gradient boosting, plus DeBERTa. Out of fold predictions are blended with non negative weights, then clipped to the range 0 to 5.
8. **Optional Gemini grader.** A rubric based score from the Gemini API, available as an extra feature through the `google.genai` SDK. It is switched off by default.

## Findings and limitations

* Ridge, SVR and gradient boosting perform similarly, and the blend is better than any single model.
* The DeBERTa model on its own was weak in this setup (cross validated RMSE about 1.02, Pearson about 0.64). It received only about 6 percent of the blend weight, so it added very little. More epochs, a tuned learning rate or a larger sample would be the first things to try.
* Predictions are pulled toward the middle of the scale at the extremes, which is typical for RMSE optimised regression on an imbalanced label distribution.
* Speech recognition errors can look like grammar errors, and Whisper sometimes silently corrects speech, which hides real mistakes.
* With only 769 training clips, the leaderboard score on a small public slice is noisy, so cross validation is the main guide.

## Data note

The competition data is **not included** in this repository. The train and test folders reuse some file names with different audio (212 test names also appear in the train folder), so the notebook resolves audio paths separately for each split. Do not merge the folders by file name.

## Repository structure

```
shl_grammar_scoring_engine/
    shl_grammar_scoring.ipynb    main notebook with code, plots and report
    submission.csv               predictions for the 216 test clips
    README.md
```

## How to run

**On Kaggle (recommended)**
1. Open the notebook on Kaggle and attach the SHL Hiring Assessment 2026 competition data.
2. Turn on a GPU accelerator and Internet access.
3. Run all cells. Transcription is the slowest step, about 20 to 30 minutes on a T4 GPU, and results are cached.
4. For the optional Gemini feature, set `USE_GEMINI = True` and add a `GEMINI_API_KEY` secret.

**Locally**
1. Download the competition data and set `ROOT` in the first data cell to the folder that contains `train.csv`, `test.csv`, `train/` and `test/`.
2. Install the dependencies:

```
pip install faster_whisper sentence_transformers language_tool_python google_genai transformers torch librosa scikit_learn pandas numpy scipy matplotlib seaborn sentencepiece
```

3. A GPU is strongly recommended. LanguageTool needs Java installed.

## Possible improvements

* Larger Whisper model for cleaner transcripts.
* Longer or better tuned DeBERTa training, or a different text encoder.
* Fine tuning wav2vec2 or the Whisper encoder directly on the audio.
* The Gemini rubric grader as an extra feature.
* Averaging over several cross validation seeds.

## Acknowledgements

Competition and data by SHL Labs, hosted on Kaggle. Built with Whisper, wav2vec2, GPT 2, sentence transformers, DeBERTa, LanguageTool and scikit learn.
