#!/usr/bin/env bash
#
# Merge Script for PR #1 - QSOL UFF v1.1.0
# This script merges the copilot/add-copilot-instructions branch into main
# and tags the release as v1.1.0 for Zenodo synchronization.
#
# Usage: ./merge_pr1.sh
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QSOL UFF - Merge PR #1 (v1.1.0)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Check if we have uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    echo "Please commit or stash your changes before proceeding."
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking out main branch...${NC}"
git checkout main

echo -e "${YELLOW}Step 2: Pulling latest changes from origin/main...${NC}"
git pull origin main

echo -e "${YELLOW}Step 3: Merging copilot/add-copilot-instructions...${NC}"
git merge copilot/add-copilot-instructions -m "Merge PR #1 — Add Copilot instructions and reproducible environment setup (v1.1.0)."

echo -e "${YELLOW}Step 4: Pushing merged main branch to origin...${NC}"
git push origin main

echo -e "${YELLOW}Step 5: Creating tag v1.1.0...${NC}"
git tag -a v1.1.0 -m "Add Copilot instructions and reproducible environment."

echo -e "${YELLOW}Step 6: Pushing tag v1.1.0 to origin...${NC}"
git push origin v1.1.0

echo -e "${YELLOW}Step 7: Verifying tag was pushed...${NC}"
if git ls-remote --tags origin | grep -q v1.1.0; then
    echo -e "${GREEN}✓ Tag v1.1.0 successfully pushed${NC}"
else
    echo -e "${RED}✗ Tag v1.1.0 not found on origin${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Merge Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary:"
echo "  • Branch: copilot/add-copilot-instructions merged into main"
echo "  • Tag: v1.1.0 created and pushed"
echo "  • Zenodo: Webhook should be triggered automatically"
echo ""
echo "Next steps:"
echo "  1. Verify the merge on GitHub"
echo "  2. Check Zenodo dashboard for new version"
echo "  3. Update documentation if needed"
echo ""
