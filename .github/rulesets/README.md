# Branch protection rulesets (snapshots)

This directory holds **exported snapshots** of the GitHub branch rulesets that
protect this repository. They are kept here for version history, auditability,
and disaster recovery (e.g. recreating protection on a new or restored repo).

## Important

- These files are **not** applied automatically. GitHub does not read rulesets
  from the repository; protection is configured in **Settings → Rules →
  Rulesets**. A snapshot here is documentation/backup only.
- After changing a ruleset in the GitHub UI, **re-export it** and update the
  corresponding file here so the snapshot stays current.

## Files

- `main_branch_protection.json` — protection for the `main` branch.

## Restoring / reusing a ruleset

1. Go to **Settings → Rules → Rulesets**.
2. Use the **New ruleset ▾ → Import a ruleset** option.
3. Select the JSON file from this directory.
4. Review and create. The `id` and `source` fields are ignored on import.

## What `main_branch_protection.json` enforces on `main`

- Require a pull request with **1 approving review**
- **Require review from Code Owners** (see `.github/CODEOWNERS`)
- Dismiss stale approvals when new commits are pushed
- Require all review threads resolved before merge
- **Squash-only** merges
- Require **linear history**
- Block force-pushes and restrict direct updates to `main`
- **Admins bypass** (so the repository owner can still administer and merge)
