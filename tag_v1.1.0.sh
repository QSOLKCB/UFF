#!/usr/bin/env bash
#
# tag_v1.1.0.sh
# Script to tag and push v1.1.0 to origin after PR #2 merge
#
# Usage: ./tag_v1.1.0.sh
#
# Prerequisites:
# - PR #2 must be merged into main
# - CHANGELOG.md must exist on main
# - User must have write access to QSOLKCB/UFF
#

set -euo pipefail

# Configuration
VERSION="v1.1.0"
TAG_MESSAGE="Copilot-integration complete; next minor release"
TARGET_BRANCH="main"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo_error "Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo_error "You have uncommitted changes in your working directory"
        echo_error "Please commit or stash your changes before running this script"
        git status --short
        exit 1
    fi
else
    echo_warn "Empty repository detected (no HEAD). Skipping uncommitted changes check."
fi

# Check if on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]; then
    echo_warn "Not on $TARGET_BRANCH branch (currently on: $CURRENT_BRANCH)"
    read -p "Switch to $TARGET_BRANCH? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo_info "Checking out $TARGET_BRANCH..."
        git checkout "$TARGET_BRANCH"
    else
        echo_error "Script requires $TARGET_BRANCH branch. Exiting."
        exit 1
    fi
fi

# Pull latest changes
echo_info "Pulling latest changes from origin/$TARGET_BRANCH..."
git pull origin "$TARGET_BRANCH"

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
    echo_error "CHANGELOG.md not found on main branch"
    echo_error "Please ensure CHANGELOG.md has been added to main before tagging"
    exit 1
fi

echo_info "CHANGELOG.md found ✓"

# Check if tag already exists locally
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo_warn "Tag $VERSION already exists locally"
    echo_info "Deleting existing local tag..."
    git tag -d "$VERSION"
fi

# Check if tag exists on remote
if git ls-remote --tags origin | grep -q "refs/tags/$VERSION"; then
    echo_error "Tag $VERSION already exists on remote"
    echo_error "If you need to re-tag, first delete the remote tag with:"
    echo_error "  git push origin :refs/tags/$VERSION"
    exit 1
fi

# Create the tag
echo_info "Creating tag $VERSION..."
git tag -a "$VERSION" -m "$TAG_MESSAGE"

# Show tag details
echo_info "Tag details:"
git show "$VERSION" --no-patch

# Verify tag creation
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    echo_info "Tag $VERSION created successfully ✓"
else
    echo_error "Failed to create tag $VERSION"
    exit 1
fi

# Push main branch
echo_info "Pushing $TARGET_BRANCH branch to origin..."
if git push origin "$TARGET_BRANCH"; then
    echo_info "$TARGET_BRANCH branch pushed successfully ✓"
else
    echo_error "Failed to push $TARGET_BRANCH branch"
    echo_error "You may need to manually push: git push origin $TARGET_BRANCH"
    exit 1
fi

# Push the tag
echo_info "Pushing tag $VERSION to origin..."
if git push origin "$VERSION"; then
    echo_info "Tag $VERSION pushed successfully ✓"
else
    echo_error "Failed to push tag $VERSION"
    echo_error "You may need to manually push: git push origin $VERSION"
    exit 1
fi

# Verify tag on remote
echo_info "Verifying tag on remote..."
if git ls-remote --tags origin | grep -q "refs/tags/$VERSION"; then
    echo_info "Tag $VERSION verified on remote ✓"
    git ls-remote --tags origin | grep "$VERSION"
else
    echo_warn "Could not verify tag on remote"
fi

# Success summary
echo ""
echo_info "========================================="
echo_info "  $VERSION Release Complete!"
echo_info "========================================="
echo ""
echo_info "Next steps:"
echo "  1. Check Zenodo dashboard: https://zenodo.org/"
echo "  2. Verify webhook triggered (5-10 minutes)"
echo "  3. New DOI will be generated automatically"
echo "  4. Update README badges if needed"
echo ""
echo_info "Concept DOI (all versions): 10.5281/zenodo.17510648"
echo_info "See RELEASE_NOTES_$VERSION.md for detailed verification steps"
echo ""

exit 0
