# Security Policy

## Scope

This is a portfolio/demonstration project (predictive maintenance ML
pipeline + API), not a released product with versioned support — security
fixes are applied to the `main` branch only.

## Reporting a Vulnerability

If you find a security issue (e.g. an auth bypass, injection vector, or a
vulnerable dependency not already tracked), please report it privately by
emailing **praffulvyas07@gmail.com** rather than opening a public issue.

Include:

- A description of the issue and its potential impact.
- Steps to reproduce, or a minimal proof of concept.
- Any suggested fix, if you have one.

You can expect an initial response within a few days. Once a fix is
available it will be merged to `main`; there is no separate disclosure
embargo process for this project.

## Known, Deliberately Deferred Issues

A few third-party dependency vulnerabilities (in `mlflow`, `starlette`, and
`cryptography`) are currently suppressed in CI's `pip-audit` step. They all
trace back to one cause: `mlflow` is pinned below 3.13 because newer
versions disable the file-based tracking backend this project's local dev
setup relies on. See the comment on `mlflow`'s pin in `pyproject.toml` and
the `pip-audit --ignore-vuln` list in `.github/workflows/ci.yml` for
details — this is tracked, not overlooked.
