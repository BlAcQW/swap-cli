# Releasing swap-cli to PyPI

Tag-triggered publish via GitHub Actions + PyPI Trusted Publishing
(OIDC). No API token sits in repo secrets.

## One-time setup (maintainer)

Done already once; documented here so a future maintainer can
re-create the PyPI publisher binding if it's ever lost.

1. Log in to https://pypi.org/ as the project owner.
2. Account → **Publishing** → **Add a new pending publisher**:
   - Project name: `swap-cli`
   - Owner: `BlAcQW`
   - Repository: `swap-cli`
   - Workflow file: `release.yml`
   - Environment: `pypi`
3. Save. After the first successful release run, the pending entry
   converts to an active Trusted Publisher and stays that way.

## Cutting a release

1. Bump the version in both files (keep them in sync):
   - `pyproject.toml` → `[project].version`
   - `src/swap_cli/version.py` → `__version__`
2. Add a new section at the top of `CHANGELOG.md` for the version +
   date + bullet list of user-visible changes.
3. Commit: `chore(release): vX.Y.Z`.
4. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```
5. Watch the `Publish to PyPI` workflow run at
   https://github.com/BlAcQW/swap-cli/actions/workflows/release.yml.
   It will:
   - Check out the tag
   - Run the test suite
   - Build wheel + sdist with `python -m build`
   - Upload to PyPI via OIDC
6. Verify the package appears at
   https://pypi.org/project/swap-cli/X.Y.Z/.
7. Draft a GitHub Release on the tag, paste the CHANGELOG section as
   the body so users see the same notes on GitHub.

## Verifying locally before release

```bash
python -m pip install --upgrade build
python -m build                                # → dist/*.whl + *.tar.gz
python -m pip install dist/swap_cli-*.whl       # in a fresh venv
swap version                                    # → matches CHANGELOG
swap doctor                                     # all rows ✓ on your box
```

## Versioning

[Semver](https://semver.org). For this stage of the product:

- **Patch** (0.1.x): bug fixes, small UX improvements, doc tweaks.
- **Minor** (0.x.0): new features, behavior changes that aren't
  breaking for existing config files.
- **Major** (x.0.0): only when we change the config schema in a
  non-migrating way, or rename the `swap` CLI itself.

## Rolling back a release

PyPI doesn't allow deleting a release once it's been pulled by anyone
(by design). If a release is broken:

1. Yank it: `python -m pip install twine && twine yank swap-cli=X.Y.Z`
   (or via the PyPI web UI). Yanking hides it from `pip install
   swap-cli` but keeps the file available for users who pin to it.
2. Fix, bump to `X.Y.Z+1`, ship that. Existing users get the fix on
   their next `pip install --upgrade swap-cli`.
