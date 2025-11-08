# Tag and Release Instructions for v1.1.0

## Overview

This document provides instructions for completing the v1.1.0 release after PR #2 has been merged. The merge includes Copilot integration, merge automation tools, and comprehensive documentation.

## Current Status

✅ **Completed:**
- PR #2 merged into main (commit: 7710b98)
- CHANGELOG.md created and ready for main
- v1.1.0 tag prepared locally
- Release documentation created

⏸️ **Pending (Requires Maintainer with Write Access):**
- Push main branch with CHANGELOG.md
- Create and push v1.1.0 tag
- Verify Zenodo webhook trigger

## Quick Start (Automated)

For maintainers with write access, use the automated script:

```bash
cd /path/to/UFF
./tag_v1.1.0.sh
```

This script will:
1. Verify you're in the correct repository
2. Checkout and pull latest main
3. Verify CHANGELOG.md exists
4. Create v1.1.0 tag with proper message
5. Push main and tag to origin
6. Verify tag on remote

## Manual Steps (Step-by-Step)

If you prefer manual execution or the script fails:

### 1. Ensure You're on Main Branch

```bash
git checkout main
git pull origin main
```

### 2. Verify CHANGELOG.md Exists

```bash
ls -la CHANGELOG.md
```

If CHANGELOG.md is missing, you need to get it from the copilot branch or recreate it.

### 3. Create the v1.1.0 Tag

```bash
git tag -a v1.1.0 -m "Copilot-integration complete; next minor release"
```

### 4. Verify Tag Creation

```bash
git tag -l v1.1.0
git show v1.1.0 --no-patch
```

Expected output should show:
- Tag name: v1.1.0
- Tagger information
- Message: "Copilot-integration complete; next minor release"
- Commit: Latest commit on main

### 5. Push Main Branch

```bash
git push origin main
```

### 6. Push the Tag

```bash
git push origin v1.1.0
```

### 7. Verify Tag on Remote

```bash
git ls-remote --tags origin | grep v1.1.0
```

Expected output:
```
<commit-sha>	refs/tags/v1.1.0
<commit-sha>	refs/tags/v1.1.0^{}
```

## Zenodo Integration Verification

After pushing the tag, Zenodo should automatically create a new version with a DOI.

### Step 1: Check GitHub Webhook Delivery

1. Go to https://github.com/QSOLKCB/UFF/settings/hooks
2. Click on the Zenodo webhook
3. Click "Recent Deliveries"
4. Find the delivery for the v1.1.0 tag push
5. Verify response status: 200 OK

### Step 2: Check Zenodo Dashboard

1. Visit https://zenodo.org/
2. Log in with appropriate credentials
3. Navigate to your uploads/deposits
4. Find "QSOL UFF" deposit
5. Look for new v1.1.0 version (may take 5-10 minutes)

### Step 3: Verify DOI Generation

Once the version appears on Zenodo:
1. Click on the v1.1.0 version
2. Note the version-specific DOI
3. Verify it's published (not draft)
4. Confirm concept DOI still points to all versions

### Expected DOIs

- **Concept DOI** (all versions): 10.5281/zenodo.17510648
- **v1.0.0**: Initial release used the concept DOI (no separate version-specific DOI)
- **v1.1.0**: Will be assigned automatically by Zenodo (format: 10.5281/zenodo.[number])

## Troubleshooting

### Issue: Tag Already Exists

**Problem:** Error message "tag 'v1.1.0' already exists"

**Solution:**
```bash
# Delete local tag
git tag -d v1.1.0

# If tag exists on remote, delete it too
git push origin :refs/tags/v1.1.0

# Recreate tag
git tag -a v1.1.0 -m "Copilot-integration complete; next minor release"
git push origin v1.1.0
```

### Issue: CHANGELOG.md Missing

**Problem:** CHANGELOG.md not found on main branch

**Solution:**
```bash
# Get CHANGELOG from the copilot branch
git checkout copilot/merge-pr2-and-update-version -- CHANGELOG.md
git add CHANGELOG.md
git commit -m "Add CHANGELOG.md for v1.1.0 release"
git push origin main
```

### Issue: Zenodo Webhook Didn't Trigger

**Problem:** Tag pushed but no new version on Zenodo

**Solutions:**
1. **Wait longer**: Zenodo can take 10-15 minutes sometimes
2. **Check webhook logs**: See GitHub webhook delivery logs
3. **Verify webhook configuration**: Settings → Webhooks → Zenodo
4. **Manual trigger**: Create new version manually on Zenodo and link to v1.1.0

### Issue: Authentication Failed

**Problem:** `git push` fails with "Authentication failed"

**Solutions:**
1. **Check credentials**: Ensure you're logged in with correct account
2. **Use SSH**: Switch to SSH URL if using HTTPS
3. **Regenerate token**: Create new personal access token with repo scope
4. **Check permissions**: Verify you have write access to QSOLKCB/UFF

### Issue: Merge Conflicts

**Problem:** Pulling main shows conflicts

**Solution:**
```bash
# Check what conflicts exist
git status

# If conflicts in CHANGELOG.md
git checkout --theirs CHANGELOG.md  # Use remote version
# OR
git checkout --ours CHANGELOG.md    # Use local version

# Resolve manually if needed
nano CHANGELOG.md

# Complete merge
git add CHANGELOG.md
git commit -m "Resolve merge conflicts in CHANGELOG.md"
```

## Rollback Procedure

If you need to rollback the release:

### Remove Tag from Remote

```bash
git push origin :refs/tags/v1.1.0
```

### Remove Tag Locally

```bash
git tag -d v1.1.0
```

### Revert Main Branch (if needed)

```bash
# Get commit hash before the problematic changes
git log --oneline -10

# Reset to that commit
git reset --hard <commit-hash>

# Force push (⚠️ DANGEROUS - use with caution)
git push --force origin main
```

## Files Included in This Release

### New Files
- `CHANGELOG.md` - Version history and release notes
- `RELEASE_NOTES_v1.1.0.md` - Detailed release notes for v1.1.0
- `TAG_AND_RELEASE_INSTRUCTIONS.md` - This file
- `tag_v1.1.0.sh` - Automated tagging script

### Files from PR #2
- `.copilot-instructions.md` - Copilot integration guide
- `MERGE_INSTRUCTIONS.md` - Merge automation documentation
- `merge_pr1.sh` - Automated merge script
- `PR1_COMPLETION_SUMMARY.md` - PR #1 completion details
- `venv_setup.sh` - Environment setup script (if not already present)

### Modified Files
- `README.md` - Updated with maintainer merge section

## Verification Checklist

Before marking the release complete, verify:

- [ ] Main branch is at correct commit (includes PR #2 merge)
- [ ] CHANGELOG.md exists on main
- [ ] v1.1.0 tag created with correct message
- [ ] Tag points to correct commit
- [ ] Main branch pushed to origin
- [ ] Tag pushed to origin
- [ ] Tag verified on remote (`git ls-remote`)
- [ ] GitHub webhook delivery successful (200 OK)
- [ ] Zenodo shows new v1.1.0 version
- [ ] Zenodo v1.1.0 version is published (not draft)
- [ ] Version-specific DOI generated
- [ ] Concept DOI still points to all versions
- [ ] README badges are up to date (optional)

## Post-Release Actions (Optional)

1. **Create GitHub Release**: Create a release on GitHub UI using v1.1.0 tag
2. **Update Documentation**: Update README badges if needed
3. **Announce Release**: Notify stakeholders/collaborators
4. **Social Media**: Share release on relevant channels
5. **Update Citation**: Add v1.1.0 DOI to citation examples

## Support and Questions

For questions or issues with this release:
- **GitHub Issues**: https://github.com/QSOLKCB/UFF/issues
- **Email**: Contact QSOL IMC team
- **Documentation**: See MERGE_INSTRUCTIONS.md, RELEASE_NOTES_v1.1.0.md

## References

- **CHANGELOG.md**: Version history
- **RELEASE_NOTES_v1.1.0.md**: Detailed release notes
- **MERGE_INSTRUCTIONS.md**: Original merge documentation
- **Zenodo Documentation**: https://help.zenodo.org/
- **Semantic Versioning**: https://semver.org/

---

**Last Updated**: November 8, 2025  
**Version**: 1.0  
**Maintainer**: QSOL IMC Team
