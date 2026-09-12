# OceanEmbed — Technical Defense Brief (plain language)

Same facts as before, pulled straight from the actual code and result files —
just explained the way you'd say it out loud to someone non-technical.

---

## 1. The three techniques

**① Transfer learning — reusing an already-trained AI instead of starting from zero**
Training an image-recognition AI from scratch needs huge amounts of data and time.
So instead, we take a **pretrained backbone** — an AI that's already been trained by
looking at a huge pile of ordinary photos (this one's called **ResNet-50**) — and we
**freeze** it: we don't let any of its trained "knowledge" change. We only add a
small **adapter** layer in front of it that reshapes our satellite readings so they
look like something that AI already knows how to read. A small **decoder** at the
end is the only part that actually learns from our ocean data.
There's also an option to swap in a more specialized satellite-trained AI from
NASA/IBM, called **Prithvi-EO**, but it needs an extra piece of software we don't
have installed, so the code automatically falls back to the plain ResNet-50 version
— and that plain version is what produced every number on your slides. Don't imply
the fancier one is what's behind your results if asked.

**② Regime conditioning (a technique called "FiLM") — telling it what kind of water it's looking at, from an independent source**
Before the AI guesses, we hand it a few extra numbers describing the general water
situation — this is the **regime signal**: is this a normally-mixed patch of ocean, a
layered **"barrier layer"** patch, or an **upwelling** patch; how **stratified**
(layered) it is; what month it is. Those numbers come only from a long-term
ocean-climate average and a river-flow estimate — **never from the same satellite
picture the model is trying to predict from.** That's what makes it **non-circular**
— the AI can't just memorise the answer by peeking. The technique that actually
injects this signal into the AI is called **FiLM** (Feature-wise Linear Modulation)
— think of it as a dial that reshapes the AI's internal signal based on the water
situation. We even have an automated check built into the code that scans this part
of the codebase every time and fails the build if it ever sneaks in a satellite
import — it's enforced by software, not just a promise.

**③ Physics-consistency loss — a physics honesty check**
While the AI is learning, we take its own guessed temperature and saltiness and
calculate how dense that water would be, using a standard ocean-science formula
called the **equation of state** (the same family as **TEOS-10**, the toolbox real
oceanographers use). If its guess would mean heavier water sitting on top of lighter
water — a **density inversion**, which can't happen in undisturbed ocean water — we
penalise it. So it's nudged toward physically sensible answers, not just numerically
close ones.

**Put together:** a patch of satellite images goes through the reused, frozen AI to
become a compressed "fingerprint." That fingerprint gets reshaped based on the
water-situation info. A small decision-making layer turns that into a full
guessed water column — temperature and saltiness at 18 depths down to 2 km. And
throughout training, the physics check keeps nudging it away from impossible
answers.
For a fair test, we also built a stripped-down version with pieces ② and ③ removed
— same AI, same data, same training — so any difference in the results is caused
only by adding those two ingredients, nothing else.

---

## 2. Where the data actually comes from

- **Ocean surface temperature** — a NOAA satellite dataset (OISST). Real, downloaded live.
- **Sea surface height** (a stand-in for currents and swirling eddies) — NOAA CoastWatch. Real, live.
- **Sea surface saltiness** — NASA's SMAP satellite. Real, live.
- **Ground truth — what's actually happening underwater** — Argo floats: free-drifting robotic instruments that sink and rise through the water measuring temperature and salt as they go. Real, live, quality-checked before use.
- **The "what kind of water is this" signal** is built from a decades-long ocean-climate average (real, live) plus a typical monthly river-flow estimate for the Ganges/Brahmaputra — this second part is **not live river-gauge data**, because that's not publicly available for these rivers; it's a well-established seasonal average instead, and the code labels it as such everywhere it's used.
- **INCOIS — India's own ocean buoy network** — we wanted this as our India-specific data source, but it wasn't accessible: their public data system couldn't be reached programmatically, and getting an official data request approved didn't happen in time. This is reported honestly as something not delivered, not glossed over — it's even logged automatically by our own fetch script when it fails.
- **Ocean colour** (would help spot plankton/sediment, i.e. river-plume water) — considered, but no usable historical satellite dataset for it was available online, so it was dropped. Saltiness partly does the same job.

---

## 3. What "Run the pipeline" actually does — and what it proves

Clicking it does **not** train a fresh AI in front of you — that would take far too
long on an ordinary laptop with no dedicated graphics card. Instead, it **replays**,
step by step, with the real timing and the real recorded results, the actual run
that happened once already on a researcher's machine. Every number and chart you
see afterward is real and was genuinely computed — just not live, while you're
watching. Say this plainly if asked; it's stated in the app itself too. Here's what
each step is actually standing in for, and what it's proving:

1. **Fetch** — downloading the raw ingredients: real satellite images and real
   Argo float readings for 2021–2023. *Proves: the whole thing runs on real,
   named public data, not invented numbers.*
2. **Match** — lining up each ocean-robot reading with the satellite picture of
   that same patch of water in that same week, then physically setting aside two
   "no peeking" test sets — the entire Bay of Bengal, and one full monsoon season
   — before any training happens. *Proves: the later test results can't be
   contaminated by the model having secretly seen the answer already.*
3. **Train baseline** — training the stripped-down version (no regime signal, no
   physics check) on the remaining data. *Proves: there's a fair, honest
   comparison point, not just OceanEmbed grading its own homework.*
4. **Train OceanEmbed** — training the full version, same data, same rules, with
   the regime signal and the physics check switched on. *Proves: the two versions
   only differ in the two ingredients being tested.*
5. **λ-physics sweep** — retrying the physics check at three different strengths.
   *Proves: the result isn't just one lucky setting — and honestly shows the
   strength barely matters at this data size, reported as-is.*
6. **Evaluate** — scoring both versions against both "no peeking" test sets and
   writing out every number the dashboard shows. *Proves: the final numbers come
   from a script anyone can re-run, not something typed in by hand.*

**What we actually achieve, end to end:** starting from nothing but what a
satellite already photographs — surface temperature, height, and saltiness — the
system reconstructs a full guess of what the water column looks like underneath,
all the way down to 2 kilometres, at 18 different depths, anywhere in the Bay of
Bengal, for any week. It does this while (a) telling you how confident it is at
every depth, (b) staying physically realistic (no impossible "heavy water floating
on light water" answers), and (c) being tested against a fair baseline on data it
was never allowed to see during training — including the honest case where it's
still slightly behind that baseline. That combination — real reconstruction, stated
confidence, physical sanity, and an honest fair test — is the actual deliverable,
not just a number going up.

---

## 4. Technical stack

**The live website** only needs a few lightweight pieces — a web-app framework
(Streamlit), a charting library, and basic spreadsheet-style tools — because it only
ever reads pre-saved result files. It never actually runs the AI while it's live.

**The heavier machinery** used to build and train the model in the first place is
separate: Python, a deep-learning framework (PyTorch), ocean-science helper
libraries, and satellite-data-download tools. It was trained on a normal laptop
with no dedicated graphics card — workable because most of the AI is frozen, and
only a small piece is actually learning.

---

## 5. Hard questions, answered plainly

**"Does it actually beat the plain version?"**
Partly — it wins during the monsoon-season test. But in the harder, more important
test — the Bay of Bengal region itself, held out entirely from training — it's
very slightly *behind* the plain version. We show this honestly instead of hiding
it; it's really in our result files, not made up to sound humble.

**"Then why bother with the physics check if it's not more accurate?"**
Because it makes the answers more physically sensible even without being more
accurate: roughly 5 times fewer "impossible water" mistakes in the hardest test,
even though both versions make very few such mistakes overall.

**"Did you use NASA's fancy satellite AI?"**
Not for these results — that's an available option in the code, but the numbers
you're seeing come from the simpler, tested version.

**"Did you get India's own INCOIS buoy data?"**
No — confirmed unreachable, and we say so rather than claim it.

**"Is the river data live?"**
No — a typical seasonal average, not live gauge readings, because live data for
these rivers isn't publicly available.

**"How do you prove it can't cheat by peeking at the data it's predicting?"**
There's a small automated check built into our code that scans that part of the
program every time and stops the build if it ever tries to use satellite data
there.

**"Why only 3 satellite readings, not 4?"**
The 4th one (ocean colour) had no usable historical satellite data available
online, so it was left out.

**"How big is the test?"**
A modest but real few thousand test points — not huge, and we say so rather than
imply otherwise.

**"Why freeze most of the AI instead of training all of it?"**
Deliberate choice, so it can actually be trained on an ordinary computer with no
graphics card, in a reasonable amount of time.

**"Is this real-time?"**
No — never claimed. Everything runs on downloaded historical files.

**"How confident is it in its own guesses?"**
It's trained to also estimate how sure it is at every depth, and we separately
double-check that by running the same guess many times with random internal
pieces switched off, and seeing how much the answer wobbles.
