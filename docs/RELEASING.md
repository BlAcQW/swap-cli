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
5. The **same tag** fires two workflows in parallel:
   - `Publish to PyPI` (`release.yml`) —
     https://github.com/BlAcQW/swap-cli/actions/workflows/release.yml.
     Checks out the tag, runs the tests, builds wheel + sdist with
     `python -m build`, uploads to PyPI via OIDC.
   - `Build Windows exe` (`build-windows.yml`) —
     https://github.com/BlAcQW/swap-cli/actions/workflows/build-windows.yml.
     On `windows-latest`, installs the base package + PyInstaller, builds a
     onefile `swap-cli.exe` from `packaging/swap-gui.spec`, and attaches
     `swap-cli-vX.Y.Z-windows-x64.exe` to the tag's GitHub Release.
6. Verify:
   - The package appears at https://pypi.org/project/swap-cli/X.Y.Z/.
   - The `.exe` is attached to the release at
     https://github.com/BlAcQW/swap-cli/releases/tag/vX.Y.Z.
7. Draft/confirm the GitHub Release on the tag, paste the CHANGELOG section as
   the body so users see the same notes on GitHub. (`softprops/action-gh-release`
   creates the release if it doesn't exist yet when it attaches the exe — edit
   it to add the notes.)

### The Windows exe

- The build is defined by `packaging/swap-gui.spec` (onefile, `console=True`,
  launches the GUI via `packaging/entry_gui.py`). The `voice` extra is excluded
  so the exe stays small.
- It is **built on Windows only** — PyInstaller does not cross-compile, so this
  cannot be produced from the Linux/Mac dev box. To smoke-test the spec without
  cutting a tag, run the workflow manually (Actions → *Build Windows exe* →
  *Run workflow*); the exe lands as a run artifact instead of a release asset.
- First proven run: flip `console=True` → `False` in the spec once you've
  confirmed the GUI opens cleanly, so customers don't see a console window.

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
