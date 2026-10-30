# Toucanet Option B



Bioacoustic machine-learning pipeline for detecting simulated human disturbance from AudioMoth recordings collected in Costa Rican rainforest environments.



This repository documents the progression of the Toucanet Option B analysis from exploratory acoustic analysis and embedding-based modeling through corrected validation, temporal modeling, and cross-site generalization experiments.



## Project Overview



The project investigates whether acoustic machine learning can detect simulated human/poacher disturbance in passive rainforest audio.



The analysis pipeline has included:



- AudioMoth field recordings

- Audio segmentation into short clips

- Perch acoustic embeddings

- Acoustic feature engineering

- Gradient Boosting and other classifiers

- Rain filtering

- Time-of-day and session confound analysis

- Corrected cross-validation to prevent data leakage

- Adaptive normalization

- Temporal sequence voting

- Cross-site / leave-one-site-out evaluation

- Alternative representation and domain-generalization experiments

- Soundscape (environmental-response) tests on the full audio, with placebo-day comparisons



A major focus of the later experiments is distinguishing true disturbance-related acoustic information from recorder/site-specific environmental information.



## At a glance

- **Question:** can acoustic machine learning detect simulated human disturbance in passive rainforest audio?
- **Early results:** within a single recorder and session, classifiers on Perch embeddings and hand-built acoustic features appeared to separate disturbance from baseline clips.
- **What the validation showed:** much of that performance was explained by site, recording session and time of day. Scripts 31-32 re-do the validation with leakage corrected, and 36-39 test the session and date confounds directly. Treat results from before script 31 with caution.
- **Where it is hard:** transfer to a recorder the model has not seen (leave-one-site-out, scripts 43, 53, 62-63). Site information is encoded in the representation itself, which is why later work moves to adaptive normalisation, sequence models and domain-adversarial methods.
- **Fall 2026 (scripts 70-81):** tests whether the soundscape itself reacts to people. BirdNET detections show no measurable silence; the full audio is slightly louder for about an hour after visits, but part of that looks like unlabelled human noise, so an animal response is not yet established. Start with 74, 77, 79 and 81.
- **Reading order for a quick review:** 31 (corrected CV), 36 and 39 (confound tests), 42-43 (sequence voting), 61 (autoregressive prediction-error model), 63 (leave-one-site-out).

Headline numbers for scripts 01-67 are not summarised in this file; see the output of each script. The section on scripts 70-81 below includes their key numbers.

## Script Guide



### 01-14 — Initial exploration and disturbance identification



Initial clip exploration, segmentation, embedding generation, clustering, metadata analysis, disturbance discovery, and simulation identification.



### 15-29 — Recorder-specific modeling and feature development



Development of AudioMoth-specific analyses, time-of-day controls, acoustic features, precision/recall evaluation, balance experiments, and analyses across additional recorder sites.



### 30 — Updated AudioMoth 1 analysis



`30_audiomoth1_analysis.py`



Updated AM1 analysis used as the starting point for the corrected validation work.



### 31-32 — Corrected cross-validation / leakage correction



`31_corrected_cv_am4.py`  

`32_corrected_cv_all.py`



Corrected validation scripts introduced after identifying leakage/confounding concerns in earlier experiments. These scripts are particularly important when interpreting results presented in later project updates.



### 33-35 — Balance, clip features, and rain filtering



- `33_corrected_balance_experiement.py`

- `34_clip_level_features.py`

- `35_rain_filter.py`



Experiments examining class balance, clip-level acoustic features, and filtering of rain-related acoustic conditions.



### 36-47b — Cross-site generalization and confound experiments



Includes:



- Cross-site training

- Session-confound analysis

- Feature ablation

- Simulation/baseline overlap checks

- Matched-date testing

- AM6 false-alarm analysis

- Hour-restricted evaluation

- Temporal sequence voting

- Leave-one-domain-out sequence voting

- Ordered-segment analysis

- Generalization experiments

- MLP/domain-adversarial approaches

- DANN leave-one-domain-out experiments



### 48-54 — Temporal and adaptive deployment experiments



Includes:



- Temporal holdout

- Adaptive z-score normalization

- Anomaly detection

- Ecological features

- Adaptive sequence voting

- No-leak adaptive normalization

- Cross-site deployment experiments

- Few-shot adaptation



### 55-57 — Numbering gap



Scripts 55-57 are not present in the current repository.



The original experiment numbering has been preserved so that script numbers continue to correspond with project notes and presentations.



### 58-60b — Representation and embedding experiments



Includes:



- Mixture of Experts

- Representation analysis

- Alternative acoustic embeddings

- CLAP embedding experiments



These experiments investigate whether cross-site limitations arise primarily from the classifier or from site information encoded in the acoustic representation.



### 61-67 — Later sequence and generalization experiments



Includes:



- Autoregressive sequence modeling

- BirdNET leave-one-domain-out analysis

- Walking-only leave-one-domain-out analysis

- GMM transition modeling

- Temporal velocity features

- Velocity plus adaptive z-score features

- Unsupervised fusion



### 70-81 — Forest response: does the soundscape react to people? (Fall 2026)

Goal: detect human presence from the **environment's response** (birds, insects, overall soundscape), not from hearing the people themselves.

All tests compare each simulation visit with **placebo windows**: the same recorder at the same clock time on days with no visit within 2 hours. This controls for the normal daily rise and fall of activity. From script 74 on, all work uses `bioacoustics_dataset.csv` (every 3-second clip, about 631k, including clips with no bird detected); scripts 70-72 are early versions on `cleaned_df` and are superseded.

**BirdNET-based tests (spreadsheet only, run locally)**

- `70_forest_response.py`: first before/during/after comparison of bird activity around visits, vs other days. Superseded by 74.
- `71_daytime_species_check.py`: daytime-only species richness check with placebo windows and recorder-running checks. Superseded by 74-75.
- `72_am6_audio_check.py`: early audio-index check at recorder 6 (bird-band loudness, ACI, entropy). Superseded by 77.
- `73_compare_label_files.py`: compares the simulation labels in the two metadata files and the visit sessions each implies. The fall analyses use `bioacoustics_dataset.csv`.
- `74_forest_response_full.py`: share of clips with a bird and species heard, before vs during vs after each visit, against placebo windows, all six recorders. **Result: no change** (27 visits with complete recordings, 29 placebo windows; e.g. bird share during−before −0.007 vs −0.005 on placebo days, p = 0.64).
- `75_species_and_power.py`: pre-specified species (shy ground birds expected quieter, alarm callers expected louder) with Holm correction, plus the smallest detectable effect. **Result: nothing survives correction; with this many visits only a drop of about half the normal bird activity (≈8 percentage points) could be detected.**
- `76_all_clips_model.py`: one fixed-effects model over every minute of recording (recorder × day and recorder × hour), with placebo-shifted sessions for inference; tests during and 0-60 min after. **Result: no significant change** (during −12% of average, placebo p = 0.56).

**Soundscape tests on the full audio (run in Google Colab; read clips from the shared Drive)**

- `77_soundscape_colab.py`: measures the sound itself in every 3-s clip around each visit and placebo window: overall, low (50-1500 Hz), bird (2-8 kHz) and high (8-16 kHz) band loudness, NDSI, ACI and spectral entropy; Holm correction over the 7 measures. **Result (27 visits, 29 placebo, all six recorders): after a visit the soundscape is louder** (overall +2.7 dB vs −0.4 on placebo days, louder after 23 of 27 visits; bird band +2.2 vs −0.9 dB; higher at 5 of 6 recorders), **but not significant after correction** (corrected p = 0.17). During visits the low band rises (corrected p = 0.04), as expected from the people themselves.
- `78_soundscape_timecourse.py`: 5-minute time course from 20 min before to 20 min after (reuses 77's measurements). No difference before visits; the difference builds from the end of the visit onwards.
- `79_soundscape_60min.py`: extends the after-window to 60 min (visits followed by another visit within the hour are dropped). **Result: the difference does not fade; it keeps growing over the hour** (bird band +2.5 dB vs −0.4 at 0-10 min, +6.3 vs −2.3 at 50-60 min), and the low band stays slightly raised for the whole hour (corrected p = 0.049).
- `80_busy_vs_quiet_hour.py`: splits visits by whether a simulation ran at any other recorder during the hour after. **Result: the rise is larger when nothing ran elsewhere**, and smaller the more activity there was elsewhere, so it is not the team's simulations at other recorders.
- `81_matched_and_listen.py`: fairest test: each visit vs only its own placebo windows (same recorder, clock time and length); also copies listening clips. **Result: about +2 dB in the bird band after visits, 11 of 16 visits (Wilcoxon p = 0.044).** Of the four "quiet-hour" visits, two show a broad rise across all bands including the low band (consistent with people, vehicles or wind) and one shows a bird-band-only rise.
- `82_noise_filtered.py`: removes "noisy" clips (low band more than 3 dB above the window's own before level) and re-runs the matched test. **Result: noisy clips are rare (~2%) and no more common after visits; with them removed the bird band is still +2.9 dB after visits, 11 of 15 visits (p = 0.041).** Listening to the clearest quiet-hour visits: birds, insects and the river, no people.
- `83_species_after_visits.py` (runs locally on the CSV): which BirdNET species change in the hour after visits, each visit vs its own normal days, FDR-corrected; plus walking-only minutes during visits. **Result: no species stands out (all q ≥ 0.72; Riverside Wren up in 7 of 8 visits is the closest hint); overall bird detections barely change; no change during walking-only minutes.** So the extra bird-band sound is not more BirdNET-detected birds.
- `84_frequency_fingerprint.py`: average spectrum in 500 Hz steps (0-16 kHz) before vs after visits, matched to normal days, noisy clips removed, to tell insects (narrow peaks) from more bird calling (broad 2-8 kHz rise) from water or wind (low-frequency rise).

**Current reading:** BirdNET detections show no measurable silence when people pass, and no species reliably increases. The full-audio soundscape is about 3 dB louder in the 2-8 kHz band for roughly an hour after visits, consistently across recorders, and this survives removing noisy clips; the clips sound like birds, insects and the river, not people. A forest response is promising but not established. Script 84 identifies which frequencies carry the rise.

Outputs from 77-81 are in `toucanet_soundscape/` (listening clips are not committed).



## Key Validation Consideration



Early experiments revealed that acoustic classification can be strongly affected by site, recording session, time of day, and related environmental differences.



For this reason, later scripts emphasize stricter validation strategies designed to reduce leakage and test whether models generalize beyond the conditions represented in their training data.



In particular, scripts **31 and 32** contain the corrected cross-validation work referenced in project presentations.



## Cross-Site Generalization



A major research question in this project is whether disturbance signatures learned at one AudioMoth location transfer to other recording locations.



Multiple approaches have been explored, including pooled training, adaptive normalization, sequence voting, domain-adversarial learning, few-shot adaptation, alternative embeddings, mixture-of-experts models, and unsupervised approaches.



Results indicate that site-specific acoustic structure is an important challenge for cross-site transfer. Later experiments therefore focus increasingly on understanding and reducing this domain dependence rather than relying solely on within-site classification performance.



## Repository Notes



This repository primarily contains analysis and experiment code.



Large audio datasets, intermediate feature matrices, embeddings, and other generated data files are intentionally not included in the repository.



Some scripts may therefore require local dataset paths or generated intermediate files before they can be executed.



## Project



**Team Toucanet — Kashmir World Foundation**



Bioacoustic AI research for wildlife conservation and detection of human disturbance in rainforest environments.

