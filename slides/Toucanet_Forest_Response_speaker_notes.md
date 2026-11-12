# Toucanet Forest Response: speaker notes

## Slide 1: Does the forest react to people?

This deck covers my fall work on Sammy's question: can we detect people from the way the forest responds, rather than from hearing the people themselves, ideally before they arrive? Everything here uses the existing recordings from the six AudioMoths; no new data was collected. I'll go through the tests step by step, then the three newest results in more detail, and finish with where this leaves us and what would settle it.

## Slide 2: What I tested, step by step

This slide walks through what I tested, in order, and what each step showed.

Step 1, the data. I use bioacoustics_dataset.csv, which Sammy confirmed: every 3-second clip, about 631,000. Following Daan's note, I line every visit up on the single moment it was logged in the field, because the labelled windows were added afterwards.

Step 2, BirdNET on all six recorders. Birds were heard about as often around visits as at the same clock time on normal days. With this many visits, a drop of about half or more is ruled out.

Step 3, the raw audio. The forest was louder after 23 of 27 visits, at five of six recorders, but not significant once corrected for the number of measures.

Step 4, what that extra sound is. It is mostly a narrow insect chorus around 12 to 13 kilohertz, and it follows the weather: those hours were warmer and drier.

Step 5, the exact logged timing. The 6 to 8 kilohertz band is a little quieter in the 25 minutes before people are logged, in 9 of 15 visits, p = 0.07. A hint, not a result.

Step 6, a gunshot check, to see whether our measures can see any animal reaction at all. They clearly pick up the played sounds and the people, but show no clear reaction from birds or insects.

Step 7, a model that cannot hear people. Using only sound above 2 kilohertz, with every clip tagged as voice, engine or dog removed, a simple model still recognises visits it has never seen, with an AUC of 0.69.

Step 8, the wind check. What the model hears is birds and insects being quieter around visits. That quiet is already there 40 minutes before the logged moment, and visit hours were about 2 kilometres per hour windier, so wind was a possible explanation.

Step 9, the deciding test. On the same visit days, hours far from any activity are normal for birds, even though they were just as windy. So the quiet is tied to the time people are in the area, not to the day or its wind.

So far: birds and insects go quieter when people are in the area. A model that cannot hear people picks this up, and it is not explained by the weather of the day. It is strong evidence, but not yet corrected for the number of tests I have run. The next slides show the last steps in detail.

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

This is the test that separates the two explanations: is the quiet about the people, or about the day?

On each visit day I took hours far from any activity, at least 3 hours from any event at that recorder and 2 hours from any event anywhere, in daylight, and compared them with the same clock time on that visit's normal days, exactly as I did for the visits. Thirteen of the 15 visit days had such hours.

The orange bars are near the visit: birds about 1.6 decibels quieter, insects about 1.2. The grey bars are the far hours of the same days. For birds, the far hours are normal, essentially zero. For insects they are a little quieter, but not significantly.

The key point is wind. The far hours were just as windy as the hours near the visit, about 2.3 kilometres per hour above normal days. So the whole day was windy, but the birds were only quiet near the visit. Wind, or anything else about the day, cannot explain the bird quieting.

Comparing within the same day, which cancels the weather and the season, the time around the visit is quieter than the far hours in 10 of 13 visit days for birds, p = 0.01, and 9 of 13 for insects, p = 0.04.

So this is the first result that holds up against the main alternative: birds, and partly insects, go quieter when people are in the area. It also explains why the quiet starts about 40 minutes before the logged moment: the team was probably already nearby, walking in or setting up, before they logged their presence. The caveat is that I have run many tests, and this is not yet corrected for that, so I call it strong evidence, not proof.

## Slide 8: Where we are, and what would settle it

To sum up. Birds, and partly insects, go quieter when people are in the area. A simple model that hears only sound above 2 kilohertz, with human sounds removed, picks this up on visits it has never seen. And the far-hours test shows it is not the weather of the day: at other hours of the same, equally windy days, the birds are normal.

Why it is not proof yet: there are only 15 usable visits and I have run many tests, so after correction it is not formally significant. And because the quiet starts about 40 minutes before the logged moment, I can't yet tell whether it is a reaction to people already nearby or a true early warning; most likely the team was in the area before they logged their presence.

What would settle it is a cleaner field test: walking-only visits with exact logged times for when people enter the forest, reach each recorder and leave, on days with similar weather, ideally with one recorder left undisturbed as a control, and played sounds with nobody nearby.

With the data we have, the next step is to get the times the team entered the forest on each visit day. Then we can see exactly when the quiet starts relative to people arriving, which is the early-warning question Sammy asked.
