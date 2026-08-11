# Roadmap to RaceSim 1.0

**North star:** fun ML project where you can **race against trained agents** on a track that feels driveable by hand.

Not the goal (for 1.0): F1-grade tires, full multi-surface µ, or publisher sim fidelity. Physics can stay a good bicycle proxy if cars interact, overtake, defend, and feel fair enough that beating a model is satisfying.

---

## Where we are today (honest baseline)

| Area | Status |
|------|--------|
| Solo lap / best-time loop | Works — Gym env, progress reward, off-track |
| Physics | Bicycle force proxy — fine for 1.0 if characterized |
| Manual control | Exists (`racesim keyboard`) — feels near-unusable |
| Trained agents | Line-following time trial; no other cars |
| Multi-car / pack racing | Not present |
| Human vs agent race day | Not present |

**Working mental model:** best-lap time simulator with research tooling.  
**1.0 mental model:** short race sessions vs opponents (scripted → learned), human in the loop.

---

## Design principles (keep decisions cheap)

1. **Payoff first.** Every phase should move “I want to race an agent this weekend” closer, not only training curves.
2. **Physics good enough.** Harden and characterize the proxy; defer Pacejka / wheel-contact until racing is fun.
3. **Ghosts before collisions.** Multi-car awareness and line choice beat perfect contact before the contact model is real.
4. **One driver contract.** Human, heuristic, and learned drivers share the same action
   output and lifecycle. Recorded ghosts use a separate time-indexed replay contract.
5. **Ship milestones as playable demos**, not only YAMLs and eval JSON.

### Evidence rule for every phase

Before implementation starts, land a versioned acceptance fixture (for example,
`configs/acceptance/v0.4.yaml`) that freezes the exact track and vehicle config,
controller or agent-bundle digests, seed list, sample count, metrics, and numeric pass
thresholds. Performance fixtures also name the reference hardware. Human-play phases
freeze the participant count, task script, time limit, and pass threshold. Closing a
phase requires machine-readable results (plus the short human-test summary when
applicable) linked from the changelog; checklist boxes alone are not evidence.

---

## Release ladder

| Version | Theme | You can… |
|---------|--------|----------|
| **0.3** | Train quality (in progress) | Train/eval a solid solo PPO with normalize + optional BC |
| **0.4** | Playable single-car | Drive by hand, enjoy a lap, beat a timer |
| **0.5** | Ghost opponents | Race a ghost of a fixed policy on the same track |
| **0.6** | Solid multi-car | 2+ bodies, collisions/blocking, pack observations |
| **0.7** | Racing agents | Policies that overtake/defend, not only follow the line |
| **0.8** | Race product | Human vs agent session, simple UI scoreboard, race modes |
| **0.9** | Polish / reliability | Suite, camera, feel, docs, fewer sharp edges |
| **1.0** | Race day | Invite a friend: “beat this agent in a 3-lap race” |

Versions are SemVer milestones, not dates. Tag one only after its checklist and frozen
acceptance fixture pass; unfinished work remains on the next pre-release version.

---

## Phase 0 — Finish solo training foundation (0.3)

**Goal:** Reliable solo training path so later racing builds on a known-good agent.

### Checklist

- [ ] Land quality training stack: VecNormalize, LR schedules, checkpoints, live eval
- [ ] Optional expert collect + BC warm-start wired and documented
- [ ] Quality env config (richer obs / shaped rewards) vs classic 19-D A/B path
- [ ] Define a versioned agent bundle: policy, VecNormalize statistics when used,
      environment/config fingerprint, observation schema version, and training metadata
- [ ] Make train/eval/play commands load that bundle and fail clearly on incompatible
      observation schemas or missing normalization state
- [ ] Smoke train config that finishes in minutes on a laptop
- [ ] `evaluate-policy` works with VecNormalize and multi-episode seeds
- [ ] Docs: `docs/training.md` + README “how to train a baseline”
- [ ] Tests for registration, obs dim, curriculum obs consistency
- [ ] Tag / changelog **0.3.0** when branch is merge-ready

**Exit test:** On the frozen 0.3 acceptance fixture, the smoke train and policy eval
complete for every declared seed, produce a self-contained agent bundle, and reproduce
the declared observation schema and metric thresholds with zero manual file wiring.

---

## Phase 1 — Playable single-car feel (0.4)

**Goal:** Manual control is the demo people try first and don’t immediately quit.

### Feel & control

- [ ] Revisit keyboard mapping / hold-vs-toggle / rate limits so the car is learnable in 30s
- [ ] Assist modes for humans: optional steering soft-centering, brake assist, speed cap for “fun mode”
- [ ] Action smoothing / deadzones tuned separately for human vs RL if needed
- [x] Existing foundation: `R` reset, terminal HUD, camera presets, and `H` autopilot toggle
- [ ] Turn reset into a complete race-session restart without restarting the process
- [ ] Expand the existing HUD with lap time, sector, off-track state, and last lap

### Camera & feedback

- [ ] Tune and validate the existing chase/top-down/free/fixed cameras on macOS
      (`mjpython`); add cockpit only if it improves the play test
- [ ] Clear off-track / wall feedback (flash / sound optional later)
- [ ] Simple lap timer and best lap persistence (local file or results/)

### Physics-for-fun (still proxy)

- [ ] Tune one “fun” vehicle preset so throttle/brake/steer response is readable
- [ ] Document drive tips in getting-started (“if it feels floating / numb, use X config”)
- [ ] Keep open-loop physics benchmarks green so feel tweaks don’t silently break CI

**Exit test:** Under the frozen 0.4 human-test script, at least 4 of 5 first-time
players complete a clean lap within 10 minutes and at least 3 record a faster second
clean lap without operator intervention.

---

## Phase 2 — Ghost racing (0.5)

**Goal:** First “race someone” without solving multi-body collision.

### World

- [ ] Ghost car: visual + timing only (no collision), driven by a recorded,
      time-indexed trajectory; live multi-body policies wait for Phase 3
- [ ] Record best-lap / policy rollout as a replay asset (npz or trajectory npz)
- [ ] Race session: N laps and optional grid/chase offsets; one shared race clock drives
      raw finish order, gap telemetry, and leaderboard output. Any handicap-adjusted
      result is a separately labeled derived field and never replaces raw order.
- [ ] Gap telemetry: delta to ghost (shared-clock time + meters ahead/behind)

### Agent plumbing

- [ ] Common `Driver` protocol: human | heuristic | SB3 policy → normalized action
- [ ] Separate `Replay` protocol: timestamp → recorded pose/timing state
- [ ] `racesim race` (or similar) CLI: human vs ghost / policy vs ghost
- [ ] Deterministic replay asset and clock so demos reproduce exactly

### Training still solo (but race-relevant)

- [ ] Evaluate policies on **race time vs fixed ghost**, not only mean return
- [ ] Leaderboard-ish JSON: track, vehicle, agent, best race gap

**Exit test:** On the frozen 0.5 fixture, two runs of the same replay produce identical
finish order and gap output; all five scripted human attempts complete with an explicit
raw result, final time gap, and no timing ambiguity.

---

## Phase 3 — Multi-car world (0.6)

**Goal:** Other cars are real in the simulation, not only in history files.

### Simulation

- [ ] Record the 1.0 architecture decision before implementation: one shared MuJoCo
      world; one Gym ego action; internal scripted/frozen opponent drivers; full
      PettingZoo/multi-agent learning remains post-1.0
- [ ] N vehicles in that shared MuJoCo world with stable per-car IDs/state
- [ ] Start with deterministic soft contact/repulsion; consider rigid-body contact only
      after the race-rule and stress fixtures pass
- [ ] Shared track projection per car (progress, lateral, heading)
- [ ] Spawn / grid start positions; pit lane optional/later
- [ ] Episode / race termination: time limit, all finished, last car out

### Observations & API

- [ ] Relative opponent features: Δprogress, Δlat, relative velocity, closest N cars
- [ ] Mask/pad obs for variable N with fixed size for SB3
- [ ] Introduce a separately registered, versioned race environment/observation schema;
      keep `RaceSim-v0` and its solo checkpoint contract compatible
- [ ] Adapt validated solo agent bundles into frozen opponents without silently changing
      their observation or normalization contract

### Safety & fairness

- [ ] Off-track / pit / reset rules for multi-car (who ends the episode?)
- [ ] No-ghost-block vs solid contact modes (config flag)
- [ ] Stress tests: 4 cars, long race, no NaNs, FPS acceptable

**Exit test:** The frozen 0.6 stress fixture completes at least 20 seeded three-car
races with zero NaNs, identity swaps, clock disagreements, or desync; it records at
least one blocked-and-repassed event and meets the declared real-time-factor budget.

---

## Phase 4 — Agents that race (0.7)

**Goal:** Opponents are not pure racing-line zombies.

### Behaviors (bottom-up)

- [ ] **Scripted race bot:** racing line + simple block/yield rules (gap thresholds)
- [ ] **Imitation pack:** BC from multi-car heuristic trajectories
- [ ] **Ego RL vs frozen bots:** reward progress + finish place + contact penalty + time
- [ ] **Curriculum:** empty track → ghost → easy bots → aggressive bots
- [ ] Overfit check: agent still finishes clean solo laps (time-trial skill kept)

### Reward / metrics for racing

- [ ] Position / place reward (sparse or dense gap-to-rival)
- [ ] Collision / dirty-air optional later; start with contact penalty
- [ ] Overtake bonus / being-overtaken penalty (easy to game — add carefully)
- [ ] Eval suite: win rate vs bot set, mean finish place, incidents/race

### Observation ablations

- [ ] Baseline with opponent features ON vs OFF (proves multi-car obs matters)
- [ ] Minimal “awareness” feature set frozen in docs for reproducibility

**Exit test:** Across at least 100 seeded races on the frozen 0.7 track and bot set,
the trained ego's 95% win-rate confidence interval has a lower bound above 50% versus
the racing-line bot, while solo clean-lap rate falls by no more than 5 percentage
points from the frozen 0.3 baseline. Incident and overtake counts are emitted for
every race.

---

## Phase 5 — Human vs agent product loop (0.8)

**Goal:** The weekend demo: *you* vs *the model*.

### Race modes

- [ ] Time trial (keep today’s fun)
- [ ] Ghost race (Phase 2, polished)
- [ ] Grid race: human + 1–3 agents, N laps
- [ ] Difficulty ladder: rookies (slow/assist) → quality PPO → aggressive bot

### UX

- [ ] One command: `racesim race-day --track … --opponent <agent-bundle>`
- [ ] Pre-race screen: controls, difficulty, track
- [ ] Post-race summary: place, best lap, gaps, incidents
- [ ] Optional third-person “broadcast” camera for watching agent-only races
- [ ] Save/load agent packs in `artifacts/` or `results/opponents/`

### Fair fight

- [ ] Same vehicle model for human and agent (no god cars unless explicit handicap)
- [ ] Optional human assists so novices can finish races
- [ ] Optional agent speed scale / reaction delay for handicap

**Exit test:** In the frozen 0.8 install test, at least 4 of 5 first-time users go from
a clean environment to a completed agent race in under 15 minutes without operator
intervention, and all five correctly identify place, elapsed time, and final gap from
the result screen.

---

## Phase 6 — 1.0 polish (0.9 → 1.0)

**Goal:** Reliable, documented, ship-shaped fun.

### Engineering

- [ ] CI: multi-car unit tests, race session smoke, physics pad still green
- [ ] Fixed seed demos and a **scripted race day** that records a short gif/mp4
- [ ] Config reference for multi-car / race rewards updated
- [ ] Performance budget (e.g. 2 cars real-time on laptop; 4 cars offline train OK)

### Content

- [ ] 2–3 “featured” tracks tuned for racing (passing zones, not pure hairpins only)
- [ ] One default opponent pack (easy / medium / hard) checked into artifacts or downloadable
- [ ] Short video or GIF in README: human vs agent finish

### Product truth

- [ ] README positions RaceSim as **raceable agents**, not “RL env only”
- [ ] Simulation honesty box kept: still not F1 tires
- [ ] Changelog **1.0.0** with “Race against agents” as the headline

**Exit test (definition of 1.0):**

1. The 0.4 first-time-player thresholds still pass on the release candidate.
2. Across at least 100 frozen-seed races, one trained opponent beats the fixed bot set
   with a 95% win-rate confidence interval whose lower bound exceeds 50%.
3. Across at least 20 human races, the medium tier has a 25–75% human win rate, and
   every result records finish order, gaps, and incidents.
4. At least 4 of 5 strangers following the README complete a human-vs-agent race in
   under 15 minutes without operator intervention.
5. Solo evaluation remains within the frozen 0.3 regression thresholds.
6. The multi-car stress and race-session smoke suites are green on the exact release
   commit.

---

## Explicit non-goals before 1.0

Defer unless something becomes unblockers:

| Item | Why wait |
|------|----------|
| Pacejka / 4-wheel contact tires | Huge rewrite; doesn’t unlock “race AI” |
| Full multi-agent RL (all cars learning) | Fun path is ego vs frozen bots first |
| Online multiplayer | Different product; offline agents are enough |
| Weather / fuel / full strategy stack | Nice-to-have after racing basics |
| Photoreal rendering | MuJoCo + clean HUD is fine |
| Huge track catalog | Prefer 2–3 great racing tracks |

---

## Suggested order of work (when you sit down)

Weekly-ish slices that always leave something playable:

1. **Now:** finish / merge Phase 0 (quality solo train).  
2. **Next:** Phase 1 keyboard + HUD only — pure fun debt paydown.  
3. **Then:** Phase 2 ghosts — first “I raced something.”  
4. **Only then:** multi-car bodies and racing rewards (Phases 3–4).  
5. **Ship loop:** race-day CLI + opponent packs (Phase 5) and polish (Phase 6).

Rule of thumb: if a PR only improves tensorboard but not race day, it should still leave a better baseline opponent for ghost/race mode.

---

## Metrics that matter (track these, not only return)

| Metric | Why |
|--------|-----|
| Human clean-lap success rate (first 10 min) | Manual control quality |
| Best lap vs racing_line delta | Solo competence |
| Race win rate human vs each bot tier | Product fun |
| Ego win rate vs fixed bot set | Agent quality |
| Incidents per race | Fairness / collision quality |
| FPS / real-time factor with N cars | Eng playability |

---

## Open decisions (resolve in the phase acceptance fixture)

Record the choice here when made:

1. **Observation partial observability:** perfect state vs noise (partial obs can wait).
2. **Human interface:** MuJoCo viewer forever vs thin game window (viewer is OK for
   1.0 if the 0.8 human test passes).
3. **Featured tracks:** choose the 2–3 release tracks before the 0.6 acceptance fixture
   freezes racing geometry and passing zones.
4. **Agent distribution:** checked-in small bundles vs release downloads, with digests
   either way.

---

## Living checklist usage

- Work phases in order unless a later phase unblocks a blocked earlier one.  
- Check boxes in PRs / CHANGELOG notes as evidence, not vibes.  
- If scope creeps toward tire science, re-read **North star** and **non-goals**.
