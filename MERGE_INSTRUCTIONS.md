# PR #1 Merge Instructions

## Overview
This document contains the shell commands to safely merge PR #1 (Copilot instructions and environment setup) into the main branch and tag it as v1.1.0 for Zenodo synchronization.

## Prerequisites
- You must have write access to the QSOLKCB/UFF repository
- Your local repository should be up to date
- You should be in the repository root directory

## Step-by-Step Merge Process

### 1. Checkout the main branch
```bash
git checkout main
```

### 2. Pull the latest changes from origin
```bash
git pull origin main
```

### 3. Merge the feature branch for PR #1
```bash
git merge copilot/add-copilot-instructions -m "Merge PR #1 — Add Copilot instructions and reproducible environment setup (v1.1.0)."
```

**Note**: If the branches have diverged and a fast-forward merge is not possible, this will create a merge commit with the specified message. If a fast-forward merge is possible, Git may ignore the `-m` flag.

### 4. Push the merged main branch to origin
```bash
git push origin main
```

### 5. Tag the merge as v1.1.0
```bash
git tag -a v1.1.0 -m "Add Copilot instructions and reproducible environment."
```

### 6. Push the tag to origin
```bash
git push origin v1.1.0
```

### 7. Verify the tag was pushed successfully
```bash
git ls-remote --tags origin | grep v1.1.0
```

## Expected Output

After successful execution, you should see:
- The main branch updated with the Copilot instructions
- A new tag `v1.1.0` created and pushed
- Zenodo webhook triggered (check Zenodo dashboard for new version)

## Files Changed in This PR

The copilot/add-copilot-instructions branch includes:
- `.copilot-instructions.md` - GitHub Copilot integration instructions
- `venv_setup.sh` - Reproducible Python environment setup script
- All existing repository files maintained

## Rollback (if needed)

If you need to rollback the merge:
```bash
# Rollback the merge commit (use the commit hash before the merge)
git reset --hard <commit-hash-before-merge>

# Force push to origin (⚠️ use with caution)
git push --force origin main

# Delete the tag locally
git tag -d v1.1.0

# Delete the tag from origin
git push origin :refs/tags/v1.1.0
```

## Alternative: Single Command Sequence

For convenience, here's a one-liner that performs all steps:
```bash
git checkout main && \
git pull origin main && \
git merge copilot/add-copilot-instructions -m "Merge PR #1 — Add Copilot instructions and reproducible environment setup (v1.1.0)." && \
git push origin main && \
git tag -a v1.1.0 -m "Add Copilot instructions and reproducible environment." && \
git push origin v1.1.0 && \
git ls-remote --tags origin | grep v1.1.0
```

## Zenodo Integration

After pushing the v1.1.0 tag:
1. Zenodo webhook should automatically detect the new release
2. A new DOI will be generated for version 1.1.0
3. Check your Zenodo dashboard at https://zenodo.org/
4. The new version should appear within a few minutes

## Notes

- This merge includes the Copilot integration guide and environment setup improvements
- The version 1.1.0 follows semantic versioning (MINOR version bump for new features)
- All existing functionality is preserved
- No conflicts are expected as this is a straightforward addition of new files
