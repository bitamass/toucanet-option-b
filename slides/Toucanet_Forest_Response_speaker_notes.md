# Toucanet Forest Response: speaker notes

## Slide 1: Does the forest react to people?

This deck covers my fall work on Sammy's question: can we detect people from the way the forest responds, rather than from hearing the people themselves, ideally before they arrive? Everything here uses the existing recordings from the six AudioMoths; no new data was collected. I'll go through the tests step by step, then the three newest results in more detail, and finish with where this leaves us and what would settle it.

## Slide 2: What I tested, step by step

This slide walks through what I tested, in order, and what each step showed.

Step 1, the data: bioacoustics_dataset.csv, every 3-second clip, about 631,000, with each visit lined up on the moment it was logged in the field.

Step 2, BirdNET: birds were heard about as often around visits as on normal days, so there is no dramatic silence.

Step 3, the raw audio: the forest was a bit louder after visits, but that turned out to be mostly an insect chorus at 12 to 13 kilohertz that follows the weather.

Step 4, the exact logged timing: one bird pitch range was a little quieter in the 25 minutes before people were logged, in 9 of 15 visits. A hint only.

Step 5, a gunshot check: our measures pick up the played sounds and the people, but no clear animal reaction.

Step 6, a model that cannot hear people: using only sound above 2 kilohertz, with voice and engine clips removed, it still recognises visits it has never seen, with an AUC of 0.69. So something in the forest's sound differs around visits.

Step 7, what it hears: birds and insects are quieter around visits, and quieter than at hours far from the visit on the same day.

Step 8, along the route: birds at the next recorder get quieter while the team is still one stop away, in 8 of 10 visits.

Step 9, the important check: at the same moments, recorders the team did not visit were often quiet too. So much of the quiet around visits comes from the days themselves, especially the evenings, not the people.

Step 10, the cleanest test: comparing all six recorders only at the same moment, so weather, light and the evening chorus cancel, recorders within two hours of the team are about 1 decibel quieter in the insect band than the others, with a permutation p of 0.02. Birds lean the same way, about half a decibel, p = 0.09.

So the forest's response to people is real but small, and it only shows when recorders are compared with each other at the same moment.

## Slide 3: Mostly insects, not birds

This slide answers the question: what is the extra sound we hear after people pass?

I measured the spectrum, in 500 hertz steps from 0 to 16 kilohertz, in the hour after each visit and compared it with the same spot and time on that visit's normal days, with noisy clips removed. Each grey line is one visit; the orange line is the median across visits.

The clearest and most consistent change is a narrow, high-pitched band at about 12 to 13 kilohertz. It is louder after 13 of 15 visits, by up to 6 decibels. A narrow, high band like this is the typical signature of an insect chorus, such as katydids or crickets. Birds rarely sing that high.

In the bird band, 2 to 8 kilohertz, the median is only about 1 decibel higher and very patchy across visits. That matches the species check: BirdNET finds no extra birds and no species that reliably increases. One visit stands out with a very large rise above 8 kilohertz, which looks like a loud insect chorus starting after that visit; the median is not driven by it.

So the forest does sound different for about an hour after people pass, and the most consistent part of that change is insects, not birds.

I then checked the weather, because insect choruses depend strongly on temperature and humidity. In the hour after visits, visit days were warmer, drier and windier than their matched normal days, and the visits that turned warmest and driest had the biggest rise in this high band. So the after-visit insect rise is probably at least partly weather rather than a reaction to people. The more promising lead is before arrival, on the next slide.

## Slide 4: A weak early-warning hint

An important correction first: each disturbance was logged in the field as a single moment, and the labelled windows in the dataset were added afterwards, 10 minutes either side for human presence. So I re-did this analysis lined up on the exact logged moment.

Two things stand out. First, every visit starts with a person: the vehicle, chainsaw and gunshot simulations almost always came shortly after someone logged their presence, so we cannot test played sounds on their own.

Second, the timing. The grey line is the lowest band, 0 to 1 kilohertz, where voices and footsteps sit. It stays flat until the last 5 minutes before the logged moment, when it rises: that is the people becoming audible. The orange line is the 6 to 8 kilohertz band, the upper part of the bird range. It drifts about 2 to 2.7 decibels below its baseline from roughly 25 minutes before the logged moment, stays down while people are there, and recovers about 20 minutes after.

So part of the soundscape seems to quiet about 20 minutes before the people themselves can be heard. That is the early-warning idea, and it survives the corrected timing in direction. But it is weak: in no single time step do at least three quarters of visits agree, and it is not statistically significant. The Great Tinamou result from before did not hold up with this timing.

I then ran one focused test: the average 6 to 8 kilohertz change from 25 to 5 minutes before the logged moment, per visit, against its own normal days. The median drop is about 2 decibels, but only 9 of 15 visits go quieter, and the p-value is 0.07, so it is not significant. As a check on the timing, the low band rises in 11 of 15 visits in the 5 minutes before logging, which is when the people become audible. So with the current visits I cannot claim a reliable early-warning signal. A clean test needs walking-only visits with logged times, which I have asked for.

## Slide 5: The forest sounds different around visits

This slide asks the question with a model instead of loudness bands, and it is the first test where we removed the people's own sounds on purpose.

I trained a simple classifier on "people present", 0 to 10 minutes after the logged moment, versus normal days at the same recorder and clock time. It is always tested on a visit it has never seen: I leave one visit out, train on the other 14, and repeat for every visit.

The key version only hears sound above 2 kilohertz, and I removed every clip where BirdNET tagged a human voice, engine, gun or dog. So it cannot be listening to the people. That model still tells visit periods from normal days, with an AUC of 0.69, where 0.5 is chance.

The orange line is its score over time, visit minus normal days. It is already raised about 30 minutes before the logged moment, peaks around the visit, and falls back to normal by 40 minutes after. The grey band is the last 5 minutes before logging, when people become audible, so most of the rise comes before anyone can be heard.

Two cautions. Perch, the bird embedding model, did worse, probably because it is built to tell species apart rather than notice overall changes. And I ran eight tests in total; after correcting for that, the best p-value is 0.08, close but not below 0.05. The next slide shows what the model is hearing.

## Slide 6: Birds and insects go quieter

So what is the model hearing? This chart shows each pitch band from 2 to 16 kilohertz, in the hour around each visit, minus the same recorder at the same clock time on normal days. Clips with human sounds are removed.

Almost every band is quieter, by about 1 to 2.5 decibels. The darker bars are the ones where at least three quarters of visits agree, 11 to 13 of 15. They sit in two places: 2 to 7 kilohertz, the main bird range, and 10 to 13 kilohertz, where insects call. So around human presence, birds and insects go quieter. That is the kind of reaction we were looking for. Our earlier loudness tests missed it because they used wide bands and the wrong timing.

The quiet builds toward the visit: comparing each visit with itself, the middle of the visit scores higher than 40 to 30 minutes before in 12 of 15 visits. That comparison cancels anything that is true of the whole day, like that day's weather or season.

On weather: temperature and humidity at visit hours were the same as on the normal days, and after a rough adjustment for weather the effect is still there in all 15 visits. The open question was wind: visit hours were about 2 kilometres per hour windier, and wind alone can make birds and insects call less. The quiet is also already there 40 minutes before the logged moment. So the next slide asks directly whether this is about the people or about the day.

## Slide 7: Quiet near visits, not the rest of the day

This test compares the time around each visit with hours far from any activity on the same day, each against normal days at the same clock time.

The orange bars are near the visit: birds about 1.6 decibels quieter, insects about 1.2. The grey bars are the far hours of the same days: for birds essentially normal, for insects a little quieter.

Within the same day, it is quieter near the visit than at the far hours, in 10 of 13 days for birds and 9 of 13 for insects. And the far hours were just as windy, so wind alone does not explain it.

But when I looked visit by visit, the big drops, 5 to 14 decibels, are all late-afternoon visits around 5 pm, when the evening chorus builds. The far hours I compared against are mostly midday, when the soundscape is much steadier. So this comparison does not control well for the evening. The next slides check exactly that.

## Slide 8: It's mostly the 5 pm visits

Before trusting the far-hours result, I checked two things.

First, does it depend on one unusual day? I recomputed it 13 times, leaving out one visit day each time. The bird result stayed significant every time, with p between 0.005 and 0.03. So no single day drives it.

But the per-day picture, this chart, shows something important. Each bar is one visit day: how much quieter the bird band was near the visit than at the far hours of that day. The six biggest drops, 5 to 14 decibels, are all afternoon visits between 4:40 and 5:30 pm, when the evening chorus builds. The morning visits, in grey, are mostly within about 2 decibels of zero.

Second, which birds? Using BirdNET's species detections, I compared how often each species was detected near the visit versus the far hours, relative to normal days. No species stands out after correcting for testing 27 species, and overall bird detections were not lower. So the quiet is in how loud the chorus is, not in fewer detectable calls.

The fact that it is mostly evenings is why I did the next two checks.

## Slide 9: Birds at the next stop quiet on approach

This test uses the walking route to ask the early-warning question.

The team walks between recorders and logs its presence at each one. So for each visit, the previous log that day at a different recorder tells me when the team was one stop back. I used the 10 visits where that previous stop was logged between 10 and 120 minutes earlier; typically it was about 14 minutes.

At the next recorder, I measured three windows, each against the same clock time on normal days: before the team reached the previous stop, 40 to 10 minutes before it; on approach, while the team was at the previous stop; and here, once the team reached this recorder.

For birds, the orange bars, the approach window is the quietest, about 1.1 decibels below normal, in 8 of 10 visits, p = 0.02. And within each visit, approach is quieter than before in 8 of 10, p = 0.05. That is the pattern you would expect if birds react while people are still one stop away.

Insects, the blue bars, are quiet in all three windows, including before, so there is no approach effect for insects.

The caveats: only 10 visits, the minute-by-minute picture is noisy, and the 'here' window is not clearly quieter for birds, possibly because people's own noise is in those clips. So it is a lead, not a finding.

## Slide 10: Unvisited recorders were quiet too

This is the check that changed my reading of the earlier results.

At the exact moment of each visit, I looked at the other recorders that the team did not visit, meaning no team event there within 60 minutes, and compared each with its own normal days at that clock time, the same way as the visited recorder. Weather and light are the same for all recorders at that moment. Seven visits had such a recorder with usable data.

Each pair of bars is one visit: orange is the visited recorder, grey is the unvisited ones at the same time, bird band. In the morning, both are small. On the two evening visits on 20 March, the visited recorder was 9 and 16 decibels quieter than normal, but the unvisited recorders were almost as quiet, 8.5 and 13 decibels. So the big evening drop in birds was a quiet evening everywhere, not a reaction to people. Across all seven, the unvisited recorders were quieter than normal in 6 of 7.

Insects tell a different story on those two evenings: the insect chorus was 6 and 12 decibels quieter at the visited recorder than at the unvisited ones. That is a people-specific signal, but only two cases.

The lesson: comparing with normal days mixes in quiet days and evenings. The fair test is to compare recorders with each other at the same moment, which is the next slide.

## Slide 11: Near the team, the forest is slightly quieter

This is the fairest test I could do with this data. I took all six recorders on every day the team was out, every 10 minutes from 6 am to 6:30 pm, and only compared recorders with each other at the same moment. Weather, light and the evening chorus are the same for all of them at that moment, so they cancel. Each recorder's own usual daily rhythm is removed too.

Each recorder at each moment is labelled by how close in time the team was to it: arriving in 20 to 60 minutes, within 20 minutes, left 20 to 60 minutes ago, or 1 to 2 hours away, compared with recorders more than 2 hours from the team.

Every bar points down: recorders near the team in time are quieter than the others at that same moment. For insects, the blue bars, it is about 1 decibel in every group. Taking all of them together, within 2 hours of the team, insects are 1.07 decibels quieter, and when I move the team's real schedule to random other recorders 2,000 times, only about 2 percent of those fake schedules look this quiet, so p = 0.02. Birds, the orange bars, lean the same way, about half a decibel, p = 0.09.

What it does not show is a sharp change when people arrive: arriving, here and after all look alike. So it looks like a small, broad dampening while the team is in the area, rather than a clear early warning.

One caveat: I chose to pool the groups after seeing that they all point the same way, so the 0.02 is best presented as a summary of this pattern.

## Slide 12: The forest responds, but quietly

To sum up where this leaves us.

What holds up: comparing all six recorders at the same moment, so weather, light and the evening chorus cancel, recorders near the team in time are about 1 decibel quieter in the insect band, p = 0.02 from the random-schedule test, and birds lean the same way. A model that only hears sound above 2 kilohertz, with human sounds removed, also tells visit periods apart from normal ones. So people in the area do leave a mark on the forest's sound.

What does not hold up: most of the bigger quiet we saw around visits, especially the 5 pm drops, also happened at recorders the team did not visit, so that part was the days and evenings, not the people. And there is no sharp change when people arrive, so I can't claim a clear early warning; birds at the next stop quieting while the team is one stop back is a lead, in 8 of 10 visits.

What it means for a detector: the effect is small, about 1 decibel. That is too small to spot from one recorder on its own, because days and evenings vary far more than that. But it shows up when recorders are compared with each other at the same moment. So a practical approach is to watch for one recorder becoming quieter than its neighbours, in the insect and bird bands, rather than watching each recorder against its own past.

Bottom line: people in the area leave a small, consistent mark on the soundscape, strongest in the insect chorus, and a model that hears only the forest can pick it up.
