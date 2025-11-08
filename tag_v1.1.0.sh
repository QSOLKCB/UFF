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

# Check if on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo_warn "Not on main branch (currently on: $CURRENT_BRANCH)"
    echo_info "Checking out main..."
    git checkout main
fi

# Pull latest changes
echo_info "Pulling latest changes from origin/main..."
git pull origin main

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
    echo_error "CHANGELOG.md not found on main branch"
    echo_error "Please ensure CHANGELOG.md has been added to main before tagging"
    exit 1
fi

echo_info "CHANGELOG.md found ✓"

# Check if tag already exists locally
if git rev-parse v1.1.0 >/dev/null 2>&1; then
    echo_warn "Tag v1.1.0 already exists locally"
    echo_info "Deleting existing local tag..."
    git tag -d v1.1.0
fi

# Check if tag exists on remote
if git ls-remote --tags origin | grep -q "refs/tags/v1.1.0"; then
    echo_error "Tag v1.1.0 already exists on remote"
    echo_error "If you need to re-tag, first delete the remote tag with:"
    echo_error "  git push origin :refs/tags/v1.1.0"
    exit 1
fi

# Create the tag
echo_info "Creating tag v1.1.0..."
git tag -a v1.1.0 -m "Copilot-integration complete; next minor release"

# Show tag details
echo_info "Tag details:"
git show v1.1.0 --no-patch

# Verify tag creation
if git rev-parse v1.1.0 >/dev/null 2>&1; then
    echo_info "Tag v1.1.0 created successfully ✓"
else
    echo_error "Failed to create tag v1.1.0"
    exit 1
fi

# Push main branch
echo_info "Pushing main branch to origin..."
if git push origin main; then
    echo_info "Main branch pushed successfully ✓"
else
    echo_error "Failed to push main branch"
    echo_error "You may need to manually push: git push origin main"
    exit 1
fi

# Push the tag
echo_info "Pushing tag v1.1.0 to origin..."
if git push origin v1.1.0; then
    echo_info "Tag v1.1.0 pushed successfully ✓"
else
    echo_error "Failed to push tag v1.1.0"
    echo_error "You may need to manually push: git push origin v1.1.0"
    exit 1
fi

# Verify tag on remote
echo_info "Verifying tag on remote..."
if git ls-remote --tags origin | grep -q "refs/tags/v1.1.0"; then
    echo_info "Tag v1.1.0 verified on remote ✓"
    git ls-remote --tags origin | grep "v1.1.0"
else
    echo_warn "Could not verify tag on remote"
fi

# Success summary
echo ""
echo_info "========================================="
echo_info "  v1.1.0 Release Complete!"
echo_info "========================================="
echo ""
echo_info "Next steps:"
echo "  1. Check Zenodo dashboard: https://zenodo.org/"
echo "  2. Verify webhook triggered (5-10 minutes)"
echo "  3. New DOI will be generated automatically"
echo "  4. Update README badges if needed"
echo ""
echo_info "Concept DOI (all versions): 10.5281/zenodo.17510648"
echo_info "See RELEASE_NOTES_v1.1.0.md for detailed verification steps"
echo ""

exit 0
