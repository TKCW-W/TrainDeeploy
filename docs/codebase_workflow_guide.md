# Codebase Management Guide
## SoC Miniproject — Managing Claude Code's Work

---

## Core Principle

**Commits** are how you save work. **PRs** are how you control what reaches `main`.  
**CI** automates what can be tested. **PR review** is the human check for everything else.

They work together — none replaces the others.

---

## Branch Structure

```
main                          ← stable, only receives merged PRs
  └── feat/maxpool-inference  ← Claude Code works here
  └── feat/maxpool-training   ← Claude Code works here
  └── feat/zo-optimization    ← future feature
  └── feat/forward-forward    ← future feature
```

- Never commit directly to `main`
- One feature = one branch = one PR
- Branch names should match the feature (consistent with Run Wang's style, e.g. `feat/speechnet-training`)

---

## Day-to-Day Workflow

### 1. Start a new feature
```bash
git checkout main
git pull
git checkout -b feat/my-feature
```

### 2. Work on the feature (Claude Code commits here)
Claude Code commits freely to the feature branch. These commits are your working history — don't worry about their quality individually.

```bash
# Claude Code does this automatically, but you can also:
git add -u
git commit -m "Fix MaxPool gradient layout: transpose forward-input X"
```

### 3. Open a PR when the feature is ready
Go to GitHub → New Pull Request → base: `main`, compare: `feat/my-feature`

You can also open a **draft PR** early if you want CI feedback before the feature is complete. This is useful when iterating with Claude Code — push, CI runs, fix, push again.

### 4. CI runs automatically
On every push to the branch, GitHub Actions runs the pre-commit checks:
- Trailing whitespace, end-of-file newlines
- `black` formatting
- `isort` import ordering
- `flake8` linting
- SPDX license headers

CI runs on the **latest commit only**. Older commits in the branch have no badge — that's normal.

**If CI fails** (like the 4/9 you saw): fix the issues locally, commit, push. CI re-runs on the new tip.

To fix pre-commit failures locally:
```bash
pip install pre-commit
pre-commit run --all-files    # auto-fixes most issues
git add -u
git commit -m "style: apply pre-commit fixes"
```

Remaining manual fixes (flake8 errors CI can't auto-fix):
- Rename ambiguous variable `l` → `lbl_int`
- Fix docstring closing quotes placement
- Reduce `##` block comment to `#`
- Add SPDX license header to new files (copy from any existing file in the repo)

### 5. Review the diff yourself
This is what your supervisor specifically asked for. Before merging:
- Read the full diff on GitHub
- CI passing does **not** mean the logic is correct — it only checks style and formatting
- Look for: wrong gradient math, unexpected file changes, architectural decisions you disagree with, subtle bugs no test would catch

### 6. Merge
Once CI is green and you're satisfied with the diff, merge the PR into `main`.

---

## What CI Catches vs. What You Catch

| | CI (automated) | PR review (you) |
|---|---|---|
| Formatting | ✅ black, isort | — |
| Style issues | ✅ flake8 | — |
| License headers | ✅ reuse | — |
| Broken ONNX export | ✅ (if test added) | — |
| Numerical regression | ✅ (if ORT test added) | — |
| Wrong gradient math | ❌ | ✅ |
| Unexpected file changes | ❌ | ✅ |
| Architectural mistakes | ❌ | ✅ |
| Logic bugs without test coverage | ❌ | ✅ |

**To-do:** add a functional CI test (ORT numerical check) as a second workflow — see below.

---

## Adding a Functional CI Test (Recommended Next Step)

The current CI only checks style. To catch correctness regressions, add a second workflow `.github/workflows/export_test.yml`:

```yaml
name: CI • Export & Numerical Check

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  export-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install torch onnx onnxruntime numpy
      - name: Export SpeechNet (MaxPool)
        run: python speechnet_exporter.py --use-maxpool --dry-run
      - name: ORT numerical check
        run: python tests/test_ort_inference.py  # feed fixed tensor, compare to saved reference
```

This catches the class of bugs like bias-applied-twice or im2col undersizing before they reach `main`.

---

## Relation to Run Wang's Repos

This workflow mirrors how Run Wang manages TrainDeeploy and Onnx4Deeploy:
- PR #2 → SpeechNet exporter (Onnx4Deeploy)
- PR #31 → SpeechNet training test (TrainDeeploy)

If you eventually upstream your MaxPool changes to Run Wang's repos, your contribution history is already clean and reviewable. Your feature branches become the basis for PRs against his repos.

---

## Quick Reference

```
New feature        →  git checkout -b feat/name
Work / Claude Code →  commits freely to feature branch
Want CI feedback   →  open draft PR early
Feature done       →  open PR (or mark draft as ready)
CI fails           →  pre-commit run --all-files + manual flake8 fixes + push
CI passes          →  review diff yourself → merge
```
