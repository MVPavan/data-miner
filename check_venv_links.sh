#!/bin/bash
# check_venv_links.sh - Analyze link status of all packages in a venv
# Usage: ./check_venv_links.sh <venv_path> <cache_path>

set -euo pipefail

# --- Args ---
VENV_PATH="${1:-.venv}"
CACHE_PATH="${2:-$HOME/.cache/uv}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# --- Validate paths ---
SITE_PACKAGES=$(find "$VENV_PATH" -type d -name "site-packages" 2>/dev/null | head -1)
if [ -z "$SITE_PACKAGES" ]; then
    echo -e "${RED}Error: No site-packages found in '$VENV_PATH'${NC}"
    echo "Usage: $0 <venv_path> [cache_path]"
    exit 1
fi

if [ ! -d "$CACHE_PATH" ]; then
    echo -e "${RED}Error: Cache directory '$CACHE_PATH' not found${NC}"
    echo "Usage: $0 <venv_path> [cache_path]"
    exit 1
fi

echo -e "${BOLD}=================================================${NC}"
echo -e "${BOLD}  UV Venv Link Analyzer${NC}"
echo -e "${BOLD}=================================================${NC}"
echo -e "  Venv:          $VENV_PATH"
echo -e "  Site-packages: $SITE_PACKAGES"
echo -e "  Cache:         $CACHE_PATH"
echo -e "${BOLD}=================================================${NC}\n"

# --- Counters ---
TOTAL=0
HARDLINKED=0
SYMLINKED=0
COPIED=0
ERRORS=0

HARDLINKED_SIZE=0
COPIED_SIZE=0
SYMLINKED_SIZE=0

# Temp files for report
REPORT_HARDLINK=$(mktemp)
REPORT_SYMLINK=$(mktemp)
REPORT_COPY=$(mktemp)
REPORT_ERROR=$(mktemp)

trap "rm -f $REPORT_HARDLINK $REPORT_SYMLINK $REPORT_COPY $REPORT_ERROR" EXIT

# --- Find all packages (top-level dirs that look like packages) ---
echo -e "${CYAN}Scanning packages...${NC}\n"

for PKG_DIR in "$SITE_PACKAGES"/*/; do
    [ ! -d "$PKG_DIR" ] && continue

    PKG_NAME=$(basename "$PKG_DIR")

    # Skip metadata, dist-info, __pycache__, bin dirs
    case "$PKG_NAME" in
        __pycache__|*.dist-info|*.egg-info|_distutils_hack|pip|pkg_resources|setuptools|*.data)
            continue
            ;;
    esac

    TOTAL=$((TOTAL + 1))

    # Find a representative file in the package
    CHECK_FILE=$(find "$PKG_DIR" -maxdepth 1 -type f -o -type l 2>/dev/null | head -1)

    if [ -z "$CHECK_FILE" ]; then
        # Try deeper
        CHECK_FILE=$(find "$PKG_DIR" -maxdepth 2 -type f 2>/dev/null | head -1)
    fi

    if [ -z "$CHECK_FILE" ]; then
        ERRORS=$((ERRORS + 1))
        echo "$PKG_NAME (no files found)" >> "$REPORT_ERROR"
        continue
    fi

    # Get package size
    PKG_SIZE=$(du -sb "$PKG_DIR" 2>/dev/null | cut -f1)
    PKG_SIZE=${PKG_SIZE:-0}

    # Check link type
    if [ -L "$CHECK_FILE" ]; then
        SYMLINKED=$((SYMLINKED + 1))
        SYMLINKED_SIZE=$((SYMLINKED_SIZE + PKG_SIZE))
        TARGET=$(readlink -f "$CHECK_FILE")
        printf "%-40s %10s  ->  %s\n" "$PKG_NAME" "$(numfmt --to=iec $PKG_SIZE)" "$TARGET" >> "$REPORT_SYMLINK"

    else
        INODE=$(stat -c %i "$CHECK_FILE" 2>/dev/null)
        LINKS=$(stat -c %h "$CHECK_FILE" 2>/dev/null)

        if [ "${LINKS:-1}" -gt 1 ]; then
            HARDLINKED=$((HARDLINKED + 1))
            HARDLINKED_SIZE=$((HARDLINKED_SIZE + PKG_SIZE))

            # Try to find matching cache file
            CACHE_MATCH=$(find "$CACHE_PATH" -inum "$INODE" 2>/dev/null | head -1)
            CACHE_DISPLAY=${CACHE_MATCH:-"(cache entry exists, links=$LINKS)"}
            printf "%-40s %10s  inode=%-10s  %s\n" "$PKG_NAME" "$(numfmt --to=iec $PKG_SIZE)" "$INODE" "$CACHE_DISPLAY" >> "$REPORT_HARDLINK"

        else
            COPIED=$((COPIED + 1))
            COPIED_SIZE=$((COPIED_SIZE + PKG_SIZE))
            printf "%-40s %10s  inode=%-10s  links=%s\n" "$PKG_NAME" "$(numfmt --to=iec $PKG_SIZE)" "$INODE" "$LINKS" >> "$REPORT_COPY"
        fi
    fi
done

# --- Print Reports ---

if [ -s "$REPORT_HARDLINK" ]; then
    echo -e "\n${GREEN}${BOLD}HARD-LINKED packages ($HARDLINKED):${NC}"
    echo -e "${GREEN}$(printf '%.0s─' {1..80})${NC}"
    sort "$REPORT_HARDLINK"
fi

if [ -s "$REPORT_SYMLINK" ]; then
    echo -e "\n${YELLOW}${BOLD}SYM-LINKED packages ($SYMLINKED):${NC}"
    echo -e "${YELLOW}$(printf '%.0s─' {1..80})${NC}"
    sort "$REPORT_SYMLINK"
fi

if [ -s "$REPORT_COPY" ]; then
    echo -e "\n${RED}${BOLD}COPIED packages ($COPIED):${NC}"
    echo -e "${RED}$(printf '%.0s─' {1..80})${NC}"
    sort "$REPORT_COPY"
fi

if [ -s "$REPORT_ERROR" ]; then
    echo -e "\n${CYAN}${BOLD}SKIPPED ($ERRORS):${NC}"
    cat "$REPORT_ERROR"
fi

# --- Summary ---
echo -e "\n${BOLD}=================================================${NC}"
echo -e "${BOLD}  SUMMARY${NC}"
echo -e "${BOLD}=================================================${NC}"
printf "  %-20s %5d packages   %10s\n" "Hard-linked:" "$HARDLINKED" "$(numfmt --to=iec $HARDLINKED_SIZE)"
printf "  %-20s %5d packages   %10s\n" "Sym-linked:" "$SYMLINKED" "$(numfmt --to=iec $SYMLINKED_SIZE)"
printf "  %-20s %5d packages   %10s\n" "Copied:" "$COPIED" "$(numfmt --to=iec $COPIED_SIZE)"
printf "  %-20s %5d packages\n" "Skipped/Errors:" "$ERRORS"
echo -e "  ${BOLD}─────────────────────────────────────────${NC}"
printf "  %-20s %5d packages   %10s\n" "Total:" "$TOTAL" "$(numfmt --to=iec $((HARDLINKED_SIZE + SYMLINKED_SIZE + COPIED_SIZE)))"

# --- Disk savings ---
if [ "$HARDLINKED" -gt 0 ]; then
    echo -e "\n${GREEN}  Disk saved by hard-links: ~$(numfmt --to=iec $HARDLINKED_SIZE)${NC}"
    echo -e "  (Hard-linked packages share disk blocks with cache)"
fi

if [ "$COPIED" -gt 0 ]; then
    echo -e "\n${YELLOW}  Potential savings if copied → hard-linked: ~$(numfmt --to=iec $COPIED_SIZE)${NC}"
    echo -e "  Tip: Ensure venv and cache are on the same filesystem,"
    echo -e "  or run: uv sync --link-mode=hardlink"
fi

# --- Percentages ---
if [ "$TOTAL" -gt 0 ]; then
    HL_PCT=$((HARDLINKED * 100 / TOTAL))
    SL_PCT=$((SYMLINKED * 100 / TOTAL))
    CP_PCT=$((COPIED * 100 / TOTAL))

    echo -e "\n${BOLD}  Distribution:${NC}"
    echo -ne "  ["
    # Simple bar chart
    BAR_WIDTH=40
    HL_BAR=$((HL_PCT * BAR_WIDTH / 100))
    SL_BAR=$((SL_PCT * BAR_WIDTH / 100))
    CP_BAR=$((CP_PCT * BAR_WIDTH / 100))
    # Fill remainder
    REMAINING=$((BAR_WIDTH - HL_BAR - SL_BAR - CP_BAR))

    echo -ne "${GREEN}"
    printf '%.0s█' $(seq 1 $((HL_BAR > 0 ? HL_BAR : 0))) 2>/dev/null
    echo -ne "${YELLOW}"
    printf '%.0s█' $(seq 1 $((SL_BAR > 0 ? SL_BAR : 0))) 2>/dev/null
    echo -ne "${RED}"
    printf '%.0s█' $(seq 1 $((CP_BAR > 0 ? CP_BAR : 0))) 2>/dev/null
    echo -ne "${NC}"
    printf '%.0s░' $(seq 1 $((REMAINING > 0 ? REMAINING : 0))) 2>/dev/null
    echo "]"

    echo -e "  ${GREEN}█ Hard-linked: ${HL_PCT}%${NC}  ${YELLOW}█ Sym-linked: ${SL_PCT}%${NC}  ${RED}█ Copied: ${CP_PCT}%${NC}"
fi

echo -e "\n${BOLD}=================================================${NC}"
