# Toucanet forest response: how each test was done

This document explains how every test was done, in the order they were run, so you can answer "how exactly did you do that?" Script numbers match the files in the `toucanet-option-b` GitHub repo, and each one's output is in `toucanet_soundscape/`.

---

## Shared building blocks (used by almost every test)

**Data.**
- `bioacoustics_dataset.csv`, as confirmed by Sammy: every 3-second clip from the six AudioMoths, about 631,000 clips.
- Each row has a recorder, a timestamp, BirdNET species with confidences, and the simulation labels ("Sim Type", "Sim Relative Time").
- The audio itself is in the shared Drive folder "Audio Moth Data / Segmented Foldered".

**Events and timing.**
- Each disturbance was logged in the field as a single moment. The ±10-minute windows in the dataset were added later in processing.
- So every event time is reconstructed as: clip timestamp minus its "Sim Relative Time". That gives the logged moment (time 0).
- 61 logged events: 29 human presence and 32 played sounds (13 vehicle, 12 gunshot, 7 chainsaw).
- Events less than 40 minutes apart at the same recorder are grouped into one visit, anchored on the first.

**Visits used.** The tests use the human-presence visits that have full recordings and at least one usable "normal day". That gives 15 visits for most tests.

**"Normal days".**
- These are the same recorder at the same clock time on other days, with no team event at that recorder within 2 hours.
- Up to 3 per visit, picked at random with a fixed seed.
- Comparing with them removes the daily rhythm (dawn and dusk choruses) and each recorder's own character.
- Important limit, found in script 99: the recordings only cover the 5 days the team was out. So "normal days" are activity days at times when the team wasn't at that recorder, not days with no team at all.

**Sound measures.**
- For each 3-second clip: resample to 32 kHz, then compute a Welch power spectrum (1,024-point segments). Average it into 48 bands of 1/3 kHz from 0 to 16 kHz, and take log10.
- **Bird band** = 2–7 kHz. **Insect band** = 10–13 kHz.
- A window's level is 10 × the mean of the log-power over its clips, in dB.
- Differences between windows are in dB: −1 dB is about 20% less sound power.

**Removing the people's own sounds.**
- Clips where BirdNET tagged human voice, engine, gun, dog, siren, power tools, chainsaw, vehicle or fireworks (at any confidence) are dropped.
- From script 92 on, most tests also use only sound above 2 kHz, where voices and footsteps are weak.

**Sampling.** To keep Colab runs manageable, each 5-minute step uses a random sample of clips (usually 8 or 12) rather than all 100. A fixed random seed makes runs repeatable.

**Statistics.**
- Per-visit values are tested with a **sign test** (how many visits go one way) and a **Wilcoxon signed-rank test** (which also uses the size of the changes).
- With 15 or fewer visits these tests have little power, so "8 of 10" can still be p = 0.1.
- Corrections for multiple tests used **Holm** (within one script's family of tests) and **FDR** (across species).
- The final test (99) uses a **permutation test**, which is the most defensible number.

**Caching.** Every script saves its measurements to Drive (`*_cache.npz`), so a Colab disconnect loses nothing. Later scripts reuse earlier measurements.

---

## Step by step

### 1. Right data and timing (scripts 73, 89)
- **Question:** which file and which times to use?
- **How:** compared the label files, then used `bioacoustics_dataset.csv`. After Daan's note, I re-anchored everything on the logged moment instead of the start of the labelled window.

### 2. BirdNET detections (scripts 74–76)
- **Question:** do birds go silent around visits?
- **How:**
  - For each visit, the share of 3-second clips with any bird detected (confidence ≥ 0.25) and the number of species, before, during and after the visit, compared with normal-day windows.
  - Script 76 fit one model over every minute with recorder × day and recorder × hour intercepts, and used fake "placebo" visit times for the p-value.
- **Result:** no drop. With this many visits, a drop of about half the normal bird activity or more is ruled out.

### 3. Raw audio loudness (scripts 77–82)
- **Question:** does the sound level change, even if BirdNET counts don't?
- **How:**
  - 7 measures per clip: total, low (50–1,500 Hz), bird (2–8 kHz) and high (8–16 kHz) band level, NDSI, ACI and spectral entropy.
  - Visit vs normal days, Holm-corrected.
  - Follow-ups looked at the 5-minute time course (78) and a 60-minute after-window (79), and checked whether the team was busy elsewhere (80). Script 81 compared each visit only with its own normal days. Script 82 removed noisy clips: clips whose low band jumped well above normal, about 2%.
  - I also listened to clips from two visits and heard birds, insects and the river, with no people.
- **Result:** louder after visits (23 of 27), not significant after correction.

### 4. What the extra sound is (scripts 84, 87)
- **How:**
  - Spectrum in 500 Hz steps, after-visit minus normal days, per visit.
  - Then hourly weather (temperature, humidity, wind from the dataset, time-shifted so the warmest hour is mid-afternoon), compared between visit and normal-day hours.
- **Result:**
  - The rise is a narrow band at 12–13 kHz, typical of an insect chorus.
  - Those hours were warmer and drier, and the rise tracked humidity. So it's mostly weather.

### 5. Exact logged timing (scripts 89, 90)
- **How:**
  - 5-minute steps from −40 to +40 minutes around the logged moment, change from the −40 to −30 baseline, minus the same on normal days.
  - Script 90 tested one pre-chosen window: the 6–8 kHz band from −25 to −5 minutes.
- **Result:**
  - 6–8 kHz is about 2 dB quieter, in 9 of 15 visits (p = 0.07).
  - The 0–1 kHz band (people) rises only in the last 5 minutes, which shows when people become audible.

### 6. Gunshot check (script 91)
- **Question:** can our measures see any animal reaction at all?
- **How:** around each played gunshot, chainsaw and vehicle, minute by minute from −5 to +9 minutes. Each minute is the median-clip level in 8 pitch bands compared with minutes −5 to −2, plus BirdNET bird share.
- **Result:** the sounds and the people show up clearly, but there's no consistent reaction from birds or insects.

### 7. A model that can't hear people (scripts 92, 93)
- **How:**
  - **Features:** for each clip, (a) the 48-band log spectrum and (b) a Perch v2 embedding (1,536 numbers, same model as the summer work). Each was made in two versions: full sound, and only above 2 kHz (high-pass filtered audio, plus human-tagged clips dropped).
  - **Classifier:** PCA to 20 components, then standardise, then balanced logistic regression, as in the summer.
  - **Training labels:** "people present" = 0 to +10 minutes after the logged moment, against all normal-day clips.
  - **Leave one visit out:** train on 14 visits and score the held-out visit and its normal days, every 5 minutes from −40 to +40. Repeat for each visit.
  - **Score per step:** held-out visit's mean log-odds minus its normal days'. Also clip-level AUC.
  - **Script 93:** a Holm correction over the 8 tests; a "shape" test of the middle of the visit against its own edges; weather regressed out; and per-band dB differences.
- **Result:**
  - The spectrum model with only sound above 2 kHz recognises unseen visits, AUC 0.69.
  - Its score is raised from about −30 to +30 minutes.
  - The best Holm-corrected p is 0.08.
  - Perch did worse (AUC 0.57–0.59).

### 8. Wind check (script 94)
- **How:**
  - Bird (2–7 kHz) and insect (10–13 kHz) bands, −30 to +30 minutes, visit minus normal days.
  - Then: low-wind visits only, the calmer half of the visits, each visit against its closest-wind normal day, and a correlation with the wind difference.
- **Result:**
  - The quiet is already present 40 minutes before the visit and stays flat.
  - On closest-wind days the bird part evens out; the insect part stays (9 of 15).

### 9. Far hours of the same day (script 95)
- **How:**
  - On each visit day, up to two "far" times: at least 3 hours from any event at that recorder and 2 hours from any event anywhere, in daylight.
  - Each far time is compared with the same clock time on that visit's normal days.
  - Paired test: near the visit minus far, within the same day.
- **Result:** far hours are normal for birds, though just as windy. Near minus far: birds 10 of 13 (p = 0.01), insects 9 of 13 (p = 0.04).

### 10. Robustness and species (script 97, no audio)
- **How:**
  - Recomputed the near-minus-far result leaving out each visit day in turn.
  - BirdNET per species: detection rate near the visit minus normal days, minus the same for the far hours, then FDR across 27 species.
- **Result:**
  - Not driven by one day (birds: p 0.005–0.03).
  - But the six biggest drops are all visits at 16:40–17:24, near the evening chorus. Mornings show little.
  - No species passes FDR.

### 11. Along the walking route (script 96)
- **How:**
  - For each visit, the previous log that day at another recorder, 10–120 minutes earlier (typically 14). That's 10 visits.
  - Three windows at this recorder, each against normal days:
    - **BEFORE:** 40 to 10 minutes before the previous stop's log
    - **APPROACH:** from the previous stop's log to 5 minutes before this log
    - **HERE:** −5 to +10 minutes
  - Plus a time course aligned on the previous stop.
- **Result:**
  - Birds are quietest on approach: −1.1 dB, 8 of 10 (p = 0.02). Approach is quieter than before in 8 of 10 (p = 0.05).
  - Insects are quiet throughout.

### 12. Recorders the team didn't visit (script 98)
- **How:**
  - At the same moment as each visit, recorders with no team event within 60 minutes (90 left too few), up to 3 per visit.
  - Each is compared with its own normal days, then visited minus unvisited. Split into morning and afternoon.
- **Result:**
  - Unvisited recorders were quieter than normal too (6 of 7).
  - On the two evenings, birds were as quiet at the unvisited recorders. Insects were 6 and 12 dB quieter at the visited one.

### 13. Same moment, all recorders (script 99): the fair test
- **How:**
  - All 6 recorders, all 5 days, every 10-minute slot from 06:00 to 18:30. That's 1,638 recorder-slots in 322 moments, with 4 clips per slot and human-tagged clips removed.
  - **Model:** level = moment intercept (one per day × slot, which absorbs weather, light and the evening chorus) + recorder × hour intercept (each spot's own daily rhythm, learned from all days) + the effect of the team's distance in time.
  - **Distance groups:** time to the nearest team event at that recorder that day, counting human presence and played sounds:
    - **approach:** 20–60 minutes before
    - **near:** within 20 minutes
    - **after:** 20–60 minutes after
    - **mid:** 60–120 minutes
    - **far:** more than 2 hours or none that day (the reference)
  - **Fitting:** least squares after removing both sets of intercepts (alternating projections).
  - **Permutation test:** keep every event's day and time, but move the team's schedule to randomly reassigned recorders on each day, 2,000 times. The p-value is the share of fake schedules that look at least as quiet as the real one. This keeps the timing (days, evenings) and only breaks where the team was. It's the one number that accounts for chance across all our testing.
  - **Pooled:** within 2 hours (all four groups) vs far. Chosen after seeing that all groups point the same way.
- **Result:**
  - Every group is quieter than far recorders at the same moment.
  - **Pooled:** insects −1.07 dB (p = 0.017), birds −0.55 dB (p = 0.09).
  - No sharp change on arrival.

---

## Likely questions from Sammy

**"How do you know it's not the weather?"**
In the final test, recorders are only compared with each other at the same 10-minute moment, and weather and light are the same for all of them then. Each recorder's own daily rhythm is also removed. Earlier tests, which compared against normal days, did pick up weather; that's why I moved to this design.

**"How do you know it's not the people's own noise?"**
Clips with BirdNET-tagged human sounds are removed. The insect band (10–13 kHz) is far above voices and footsteps. And people's sounds would make recorders *louder*, not quieter.

**"Where does the p-value come from?"**
From moving the team's real schedule to random other recorders 2,000 times and refitting each time. p = 0.017 means about 2% of fake schedules looked as quiet as the real one.

**"Why not just BirdNET?"**
BirdNET counts detections, and those didn't change. The effect is in how loud the chorus is.

**"Is it early warning?"**
Not clearly. The quiet doesn't sharpen as people arrive. The route test (birds quieter while the team is one stop away, 8 of 10) is the best lead.

**"How big is the effect?"**
About 1 dB in the insect band, roughly 20% less sound power. That's small next to day-to-day changes, so it's only visible when recorders are compared with each other at the same moment.

**"What about Perch embeddings?"**
They did worse than the simple spectrum (AUC about 0.58 vs 0.69), probably because Perch is trained to tell species apart, not to measure overall chorus level.

**"How many visits?"**
15 human-presence visits with normal days for most tests. All 61 events (presence and played sounds) on 5 days, across 6 recorders, for the final test.
