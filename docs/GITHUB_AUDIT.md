# GitHub Audit (Coding, Security, and Project Quality)

This audit is focused on practical improvements you can make in this repository and on GitHub settings.

## What was added in this PR

1. **CI workflow** (`.github/workflows/ci.yml`)
   - Runs Ruff lint checks.
   - Runs a Python compile smoke test.
   - Runs `pip-audit` to catch vulnerable dependencies.

2. **CodeQL workflow** (`.github/workflows/codeql.yml`)
   - Adds static security analysis for Python.
   - Runs on push, PR, and weekly schedule.

3. **Dependabot config** (`.github/dependabot.yml`)
   - Weekly update PRs for Python dependencies and GitHub Actions.

## High-impact recommendations (next)

### Security hardening

- **Enable branch protection on `main`**
  - Require PRs and at least 1 review.
  - Require status checks from CI + CodeQL.
  - Disallow force pushes and branch deletion.

- **Turn on secret scanning + push protection**
  - In repository security settings, enable secret scanning alerts.
  - Enable push protection to block committing known secret formats.

- **Pin action versions by commit SHA**
  - More secure than tags (`@v4`, `@v5`) because tags can move.

- **Add a SECURITY.md**
  - Give users a clear disclosure path for vulnerabilities.

### Coding quality

- **Split `spyer.py` into modules**
  - Suggested structure:
    - `spyderscalp/ui/`
    - `spyderscalp/signals/`
    - `spyderscalp/data/`
    - `spyderscalp/options/`
    - `spyderscalp/storage/`
  - This improves testability and reduces regression risk.

- **Introduce type checking gradually**
  - Start with `mypy --ignore-missing-imports` on a few core functions.
  - Add type hints around data boundaries and settings/history serialization.

- **Reduce `except Exception: pass` usage**
  - Silent failures hide production bugs.
  - At least log errors to `CRASH_LOG` with context for diagnostics.

### Reliability

- **Add unit tests for scoring logic**
  - Test deterministic pieces first (grading thresholds, DTE choice, hold-time logic).
  - Keep market/network calls mocked.

- **Add caching/backoff metrics to logs**
  - Emit structured logs for yfinance retries and failures.
  - Helps tune scan intervals and catch API throttling issues.

### “Cooler” project polish

- **Add badges for CI, CodeQL, and Dependabot status** to README.
- **Add a demo GIF** in README showing chart + signal updates.
- **Add issue templates** for bug reports and feature requests.
- **Add PR template** with checklist (tests run, risk notes, screenshots).
- **Add releases with changelog** to make updates feel professional.

## Suggested GitHub settings checklist

- [ ] Require pull request reviews before merging.
- [ ] Require status checks to pass before merging.
- [ ] Require branches to be up to date before merging.
- [ ] Include administrators in branch protection.
- [ ] Enable dependency graph and Dependabot alerts.
- [ ] Enable Code scanning alerts.
- [ ] Enable secret scanning + push protection.

## Stretch goals

- Add `pre-commit` hooks (ruff, trailing-whitespace, end-of-file-fixer).
- Add optional telemetry (local-only, privacy-safe) to track false signal rates.
- Publish a roadmap board in GitHub Projects.
