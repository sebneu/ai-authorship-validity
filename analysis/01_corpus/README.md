# Ground-truth corpus construction

Design for the positive and negative sets underpinning RQ1–RQ3. Frozen as `corpus-v1`
once built; every count below is regenerable from `profile_aidev.py` and
`probe_crossd.py`.

The paper's core claim is that detector error is measured, not assumed. That puts the
weight on this directory: if the ground truth is contaminated or confounded, nothing
downstream survives. Each choice below is therefore recorded with the threat it closes.

---

## 1. What AIDev actually contains

Measured at revision `68ed5f4b80d2` (2026-05-10), not taken from the paper — the dataset
has roughly doubled since arXiv:2507.15003 reported 456k PRs.

| Table | Rows | Role |
|---|---|---|
| `all_pull_request` | 932,791 | PR titles + bodies, agent-labelled |
| `pull_request` | 33,596 | curated subset, 2,807 popular repos |
| `pr_commit_details` | 711,923 | commit messages **and patches** |
| `pr_commits` | 88,576 | commit messages |
| `pr_comments` | 39,122 | comment bodies, carries `user_type` |
| `issue` | 4,614 | issue titles + bodies |
| `human_pull_request` | 6,618 | human-authored PRs, same era |
| `repository` | 2,807 | language, stars, forks, license |

### Three properties that drive the whole design

**(a) The positive window is seven months.** All agent PRs fall between **2024-12-24 and
2025-07-30**. Findings are therefore about H1-2025 agent output and the model versions
current then. Temporal generalisation is a stated external-validity threat, not a claim.

**(b) Agent mix is extremely skewed.** OpenAI Codex is 814,522 of 932,791 PRs — **87%**.

| Agent | PRs | Median body |
|---|---|---|
| OpenAI Codex | 814,522 | 283 chars |
| Copilot | 50,447 | 2,681 chars |
| Cursor | 32,941 | 331 chars |
| Devin | 29,744 | 863 chars |
| Claude Code | 5,137 | 1,506 chars |

Median body length varies **~10× across agents**, and the pooled median (294 chars) is
essentially Codex's. **Any pooled detector metric is a Codex metric wearing a general
label.** All RQ1 results are reported per agent; pooled numbers appear only with
explicit equal-allocation reweighting. This is itself a finding about the prevalence
literature, which reports pooled figures.

**(c) The texts are short.** Median PR body 294 chars; median **commit message 56 chars**
(p90 164). Zero-shot detectors are calibrated on essay-length text. The commit-message
genre is where they should fail hardest — that is the hypothesis, and short-length
stratification is how it gets tested rather than asserted.

Two further facts worth carrying into RQ2:

- **39.2% of agent commits already carry a `Co-authored-by:` trailer** — the declaration
  channel is strong where it exists, corroborating Khosravani & Mockus from the other side.
- **70% of comments on agent PRs are bot-authored** (`user_type`: 27,416 Bot vs 11,706
  User). The comment genre must be filtered by `user_type` or it measures CI-bot text.

---

## 2. Sampling frame

The 2,807 curated repositories, restricted to those existing before the cutoff.

**Repo age check (25 sampled, GitHub API): 71% were created before 2022-11-30.** So the
same-repo design covers roughly **2,000 repositories**; the remaining ~29% are too young
to have a pre-LLM history at all.

Those excluded repos are not missing at random — they are newer, and newer projects
skew toward agent adoption. The exclusion is reported, and RQ1 carries a
cross-repo generalisation check on the excluded set (positives only, no matched
negatives) to show results do not hinge on repo age.

---

## 3. Positive sets

| Genre | Source | Selection |
|---|---|---|
| PR description | `all_pull_request` | non-empty `body`; use the **full** table, not the curated subset, so Claude Code has 5,137 rather than 459 rows |
| Commit message | `pr_commits` | non-empty `message`; trailer lines stripped into a separate field, never fed to detectors (they are a giveaway, and detecting them is not detection) |
| Code diff | `pr_commit_details` | `patch`, per file, language from `repository.language` |
| Issue body | `issue` | non-empty `body` |
| PR comment | `pr_comments` | `user_type == 'User'` only |

Cell counts available for the curated join, with comments already restricted to
`user_type == 'User'` (all ≥ 428, most in the thousands):

| Agent | PR body | Commit msg | Diff | Comment |
|---|---|---|---|---|
| Claude Code | 459 | 2,781 | 23,296 | 428 |
| Copilot | 4,964 | 23,813 | 187,247 | 6,932 |
| Cursor | 1,431 | 6,173 | 37,838 | 523 |
| Devin | 4,813 | 23,281 | 169,467 | 1,986 |
| OpenAI Codex | 21,569 | 32,528 | 294,075 | 1,837 |

Dropping bot comments costs more than half the comment genre — Codex falls from 6,706 to
1,837 — which is the bot-confounder problem showing up inside the positive set, before
any detector runs.

**Allocation: equal per agent, not proportional.** Target n = 1,000 per (agent × genre),
floor 300, capped by availability. Equal allocation costs precision on Codex and buys it
on Claude Code — the right trade when the question is whether detectors generalise
*across* agents. The binding cells are Claude Code comments (428) and Cursor comments
(523); both clear the floor, so no cell is dropped.

---

## 4. Negative sets — three of them

The plan specified one negative set. Three are needed, because they fail in different
directions and no single one is both pure and realistic.

### N1 — pre-LLM, same repositories (the counterfactual)

Artifacts from the ~2,000 pre-2022 frame repos, dated before **2022-11-30**. Every
detector hit here is a false positive **by construction**. This is what yields the FPR
used in the RQ3 correction.

**Purity rule — filter on the timestamp of the text, never of its parent.** An artifact
qualifies only if `created_at < cutoff` **and** `updated_at < cutoff`. Bodies are
editable after the fact, and issue threads accrue comments forever: the CrOSSD probe
found React issue #10 (created 2013) carrying a 2026 comment that opens
`## Verification Results (Agent-Executed)`. Filtering on creation date alone would ingest
2026 agent text as 2013 ground truth.

The rule is conservative — an old issue touched by a label change is dropped — which
biases N1 toward *quiescent* artifacts. That is a real confound with register and
length, so N1 is length-matched (below) and the quiescence bias is stated in threats.

Acquisition per genre:

- **Commit messages and diffs:** `git clone --filter=blob:none`, then `git log`/`git show`
  bounded by committer date. Immutable short of a history rewrite, no rate limit, no
  edit-contamination problem at all. Preferred source.
- **PR bodies, issue bodies, comments:** GitHub REST API with the double-timestamp filter.
- **Cross-check:** GH Archive event payloads for a random subsample, confirming that
  API-returned bodies match what was written at the time. This validates the
  double-timestamp rule rather than trusting it; if it fails, GH Archive becomes primary.

**Size:** FPR is the precision-critical quantity. For p ≈ 0.01 at ±0.3 pp (Wilson),
n ≈ 4,200. Target **5,000 per genre** for the headline per-genre FPR, giving five strata
at usable precision; finer stratification (language, length decile) reports wider
intervals rather than pretending to the same precision.

### N2 — contemporaneous human (realism)

`human_pull_request`: 6,618 PRs, 818 repos, 2,515 users, Jan–Jun 2025, median body 468
chars. Same era as the positives, so no temporal confound — this is the *discrimination*
task as actually faced in practice.

Purity is weaker: a human PR body may well have been drafted with ChatGPT, invisibly.
N2 therefore bounds performance from one side only. Detector errors on N2 are reported as
**apparent** FPR, and the gap between N1 and N2 error rates is itself informative — it
estimates undeclared assisted use in the human population, which is exactly the
unobservable stratum the paper is otherwise honest about not measuring.

### N3 — pre-LLM bots (the confounder)

Pre-2022 artifacts from known bot accounts (dependabot, renovate, greenkeeper, release
automation) in the same repos, identified via the BoDeGHa list plus a manual list.
Templated, machine-generated, and definitively **not** LLM output. FPR here quantifies
how much reported "AI prevalence" is old automation being re-labelled.

Given that 70% of comments on agent PRs are already bot-authored, this set is not a minor
robustness check — it is central.

---

## 5. Matching and stratification

Detectors are length-sensitive and agents differ ~10× in length, so an unmatched
comparison would measure length, not authorship.

- **Length matching:** negatives sampled to match the positive length distribution within
  genre, by decile. Unmatched results reported alongside, since the delta between matched
  and unmatched *is* the length-confound estimate.
- **Repo matching:** N1 and N3 drawn from the same repositories as the positives.
- **Language:** from `repository.language` (TypeScript 650, Python 530, Go 242, C# 220,
  JavaScript 190, Rust 159 of 2,807) — reported for diffs, where it should matter most.
- **Project scale:** stars (median 564, p10 135, p90 10,253) and the CrOSSD covariates
  for the intersection of the two panels.

## 6. Outputs

```
data/processed/corpus_v1/
  positives.parquet    id, agent, genre, repo, text, n_chars, created_at, source_table
  negatives.parquet    id, negative_set (N1|N2|N3), genre, repo, text, n_chars, ...
  manifest.json        revisions, cutoffs, seeds, per-cell counts, checksums
```

One row = one text passed to a detector. No detector ever sees repository, agent or date
fields — the harness reads `text` only, and scores join back on `id`.

## 7. Open items before freezing

1. **Run `verify_repo_ages.py`** to replace the 25-repo estimate (~71%) with the measured
   share over all 2,807, and to emit the eligible-repository list that the N1 builder
   consumes. Needs `GITHUB_TOKEN` in the environment; ~12 min at 4 workers.
2. Fix the bot-account list; decide whether GitHub's `user_type == 'Bot'` is sufficient
   or whether BoDeGHa-style classification is needed for pre-2022 accounts.
3. Confirm GH Archive cross-check passes, or promote GH Archive to primary.
4. Decide whether diffs are scored per file or per commit — DetectCodeGPT assumes
   contiguous code, and per-file is the closer match.
