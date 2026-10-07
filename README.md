\# Toucanet Option B



Bioacoustic machine-learning pipeline for detecting simulated human disturbance from AudioMoth recordings collected in Costa Rican rainforest environments.



This repository documents the progression of the Toucanet Option B analysis from exploratory acoustic analysis and embedding-based modeling through corrected validation, temporal modeling, and cross-site generalization experiments.



\## Project Overview



The project investigates whether acoustic machine learning can detect simulated human/poacher disturbance in passive rainforest audio.



The analysis pipeline has included:



\- AudioMoth field recordings

\- Audio segmentation into short clips

\- Perch acoustic embeddings

\- Acoustic feature engineering

\- Gradient Boosting and other classifiers

\- Rain filtering

\- Time-of-day and session confound analysis

\- Corrected cross-validation to prevent data leakage

\- Adaptive normalization

\- Temporal sequence voting

\- Cross-site / leave-one-site-out evaluation

\- Alternative representation and domain-generalization experiments



A major focus of the later experiments is distinguishing true disturbance-related acoustic information from recorder/site-specific environmental information.



\## Script Guide



\### 01-14 — Initial exploration and disturbance identification



Initial clip exploration, segmentation, embedding generation, clustering, metadata analysis, disturbance discovery, and simulation identification.



\### 15-29 — Recorder-specific modeling and feature development



Development of AudioMoth-specific analyses, time-of-day controls, acoustic features, precision/recall evaluation, balance experiments, and analyses across additional recorder sites.



\### 30 — Updated AudioMoth 1 analysis



`30\_audiomoth1\_analysis.py`



Updated AM1 analysis used as the starting point for the corrected validation work.



\### 31-32 — Corrected cross-validation / leakage correction



`31\_corrected\_cv\_am4.py`  

`32\_corrected\_cv\_all.py`



Corrected validation scripts introduced after identifying leakage/confounding concerns in earlier experiments. These scripts are particularly important when interpreting results presented in later project updates.



\### 33-35 — Balance, clip features, and rain filtering



\- `33\_corrected\_balance\_experiement.py`

\- `34\_clip\_level\_features.py`

\- `35\_rain\_filter.py`



Experiments examining class balance, clip-level acoustic features, and filtering of rain-related acoustic conditions.



\### 36-47b — Cross-site generalization and confound experiments



Includes:



\- Cross-site training

\- Session-confound analysis

\- Feature ablation

\- Simulation/baseline overlap checks

\- Matched-date testing

\- AM6 false-alarm analysis

\- Hour-restricted evaluation

\- Temporal sequence voting

\- Leave-one-domain-out sequence voting

\- Ordered-segment analysis

\- Generalization experiments

\- MLP/domain-adversarial approaches

\- DANN leave-one-domain-out experiments



\### 48-54 — Temporal and adaptive deployment experiments



Includes:



\- Temporal holdout

\- Adaptive z-score normalization

\- Anomaly detection

\- Ecological features

\- Adaptive sequence voting

\- No-leak adaptive normalization

\- Cross-site deployment experiments

\- Few-shot adaptation



\### 55-57 — Numbering gap



Scripts 55-57 are not present in the current repository.



The original experiment numbering has been preserved so that script numbers continue to correspond with project notes and presentations.



\### 58-60b — Representation and embedding experiments



Includes:



\- Mixture of Experts

\- Representation analysis

\- Alternative acoustic embeddings

\- CLAP embedding experiments



These experiments investigate whether cross-site limitations arise primarily from the classifier or from site information encoded in the acoustic representation.



\### 61-67 — Later sequence and generalization experiments



Includes:



\- Autoregressive sequence modeling

\- BirdNET leave-one-domain-out analysis

\- Walking-only leave-one-domain-out analysis

\- GMM transition modeling

\- Temporal velocity features

\- Velocity plus adaptive z-score features

\- Unsupervised fusion



\## Key Validation Consideration



Early experiments revealed that acoustic classification can be strongly affected by site, recording session, time of day, and related environmental differences.



For this reason, later scripts emphasize stricter validation strategies designed to reduce leakage and test whether models generalize beyond the conditions represented in their training data.



In particular, scripts \*\*31 and 32\*\* contain the corrected cross-validation work referenced in project presentations.



\## Cross-Site Generalization



A major research question in this project is whether disturbance signatures learned at one AudioMoth location transfer to other recording locations.



Multiple approaches have been explored, including pooled training, adaptive normalization, sequence voting, domain-adversarial learning, few-shot adaptation, alternative embeddings, mixture-of-experts models, and unsupervised approaches.



Results indicate that site-specific acoustic structure is an important challenge for cross-site transfer. Later experiments therefore focus increasingly on understanding and reducing this domain dependence rather than relying solely on within-site classification performance.



\## Repository Notes



This repository primarily contains analysis and experiment code.



Large audio datasets, intermediate feature matrices, embeddings, and other generated data files are intentionally not included in the repository.



Some scripts may therefore require local dataset paths or generated intermediate files before they can be executed.



\## Project



\*\*Team Toucanet — Kashmir World Foundation\*\*



Bioacoustic AI research for wildlife conservation and detection of human disturbance in rainforest environments.

