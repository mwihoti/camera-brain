# Plan: camera AI → product / research / funding

Goal: one strong artifact (tracked, measured, written-up pilot) that opens
three doors at once — sales, research community, grant applications.

## Week 1 — Make the data trustworthy (tech foundation)

- [ ] Build YOLO + ByteTrack tracker into traffic_watch.py (Claude builds)
      → unique vehicle IDs, direction, moving vs parked, dwell time
      → AI (NVIDIA/Claude) only called on NEW tracked objects (crops)
- [ ] Bring Mevo Core online via USB-C (lens on, webcam mode)
      → wire as camera #2 into the same pipeline
- [ ] Simple daily report generator: one HTML/PDF page per day
      (counts per hour chart, peak times, unique vehicles, snapshots)
- [ ] 48-hour unattended stability run — zero manual intervention allowed
      (services must self-heal; fix whatever breaks)

## Week 2 — Run one real pilot

- [ ] Pick ONE site: building gate (best), or one event (fallback)
- [ ] Put up a "CCTV/analytics in use" notice (Data Protection Act basics)
- [ ] Collect 7 continuous days of tracked data
- [ ] Daily 10-min review: note failures, false counts, surprises (this log
      becomes the "findings" section later)

## Week 3 — Turn the pilot into assets

- [ ] Write-up (blog post / long X thread): what was built, real numbers,
      failures included (hallucinating models, flaky APIs, WiFi wars)
- [ ] One-page PDF with the pilot's actual charts — the sales one-pager
- [ ] 2-minute screen-recorded demo video (live labeled view + dashboard)
- [ ] Clean the repo, add README — this is now the portfolio piece

## Week 4 — Open the three doors

- [ ] Sales: demo to 3 real prospects (estate manager, venue owner, retail)
      — offering a free 2-week pilot, not a sale
- [ ] Research: post write-up, join Deep Learning Indaba / IndabaX Kenya +
      Data Science Africa channels, contact DSAIL (Dedan Kimathi, wildlife CV)
- [ ] Funding: apply NVIDIA Inception (free, immediate) + draft Lacuna Fund
      concept note (African road/gate visual dataset angle); look at IDRC AI4D
      calls

## Standing decisions

- Local-first architecture: YOLO on CPU does constant work; cloud models
  (NVIDIA primary, Claude fallback) only see crops + write summaries.
- Data protection is a feature: auto-deletion retention, on-prem processing,
  visible notices. Say it in every pitch.
- One pilot, deeply done, beats three shallow demos.

## Progress log

- 2026-08-16: V380 unlocked (RTSP/ONVIF), live feed + recording working
- 2026-08-17: AI agents (room, traffic, attendance), PTZ control, NVIDIA
  models integrated with Claude fallback, live labeled viewer, survived
  weak-WiFi + flaky-API wars with circuit breakers
- 2026-08-20: Mevo Core acquired (4K, MFT lens, USB/SRT/NDI) — camera #2

## From prototype to fundable project (added 2026-08-20)

The formula: ONE named problem + ONE real deployment + THREE numbers.

1. Pick one problem, phrased as money: "estates lose track of who enters;
   we give managers a searchable vehicle log for less than a guard shift".
2. One real pilot site (not my desk) — free 30 days for feedback + case study.
3. Earn the three judging numbers:
   - accuracy (% vehicles logged vs 48h ground truth)
   - reliability (30 days unattended, downtime, self-recoveries)
   - unit economics (KSh per gate/month all-in)
4. The boring wrapper: 1-page data-protection statement (DPA as a feature),
   simple SLA, explicit pricing after pilot, ODPC registration later.
5. Edge box: used mini-PC (~KSh 20-25k) running the stack headless at the
   pilot site. Kills "runs on my laptop". Most important purchase.
6. Same dossier feeds every door: NVIDIA Inception -> Lacuna concept note
   (Kenyan road/gate dataset) -> AI4D/GIZ -> accelerators + 3 sales pitches.
7. Sequence: wk1-2 choose+site+box, wk3-6 pilot+numbers, wk7 case study,
   wk8 applications and pitches. Cash cost ≈ the mini-PC.
