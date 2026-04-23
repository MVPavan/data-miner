#!/usr/bin/env bash
# transfer_laion_good2.sh
#
# 3-step resumable transfer of the LAION Good2 dataset to a remote machine:
#   images : /media/data_1/deepak_cr/LaionFiltered/Good2            (~5.4M JPGs, ~1 TB)
#   labels : output/auto_annotation_v4/laion_good2/labels_standalone (~5M .txt, ~1 GB)
#
# No rsync/zstd required — uses only tar, split, gzip, ssh, scp (all present
# on this box). Resumability comes from splitting the images tar into chunks
# and scp-ing each chunk with retry + idempotent skip-if-same-size.
#
# Three steps (run them in order):
#   1.  archive   — build labels.tar.gz + images.tar.part.####  in STAGE_DIR
#   2.  transfer  — scp everything to the remote (per-chunk retry; skip done)
#   3.  extract   — reassemble + untar on the remote
#
# After step 3 succeeds, run `cleanup` to free staging disk.
#
# Usage:
#   DEST_HOST=otherbox DEST_USER=pavan DEST_DIR=/data/laion_good2 \
#     scripts_bench/transfer_laion_good2.sh archive
#   DEST_HOST=... ... scripts_bench/transfer_laion_good2.sh transfer
#   DEST_HOST=... ... scripts_bench/transfer_laion_good2.sh extract
#   scripts_bench/transfer_laion_good2.sh cleanup
#
# Do only labels OR only images with WHAT=labels / WHAT=images (default: all).
#
# Env:
#   DEST_HOST    remote hostname/IP (REQUIRED for 2/3)
#   DEST_USER    remote user        (default: current user)
#   DEST_DIR     remote target dir  (REQUIRED for 2/3)
#   SSH_PORT     ssh port           (default: 22)
#   STAGE_DIR    local staging      (default: /media/data_2/_laion_transfer_stage)
#   CHUNK_SIZE   images tar chunk   (default: 20G)
#   WHAT         labels|images|all  (default: all)

set -euo pipefail

IMAGES_SRC="/media/data_1/deepak_cr/LaionFiltered/Good2"
LABELS_SRC="/media/data_2/vlm/code/data_miner/output/auto_annotation_v4/laion_good2/labels_standalone"

DEST_USER="${DEST_USER:-${USER:-root}}"
SSH_PORT="${SSH_PORT:-22}"
STAGE_DIR="${STAGE_DIR:-/media/data_2/_laion_transfer_stage}"
CHUNK_SIZE="${CHUNK_SIZE:-20G}"
WHAT="${WHAT:-all}"

LABELS_ARCHIVE="${STAGE_DIR}/labels.tar.gz"
IMAGES_PARTS_DIR="${STAGE_DIR}/images_parts"

banner()   { echo; echo "====== $* ======"; }
need_bin() { command -v "$1" >/dev/null 2>&1 || { echo "missing local tool: $1" >&2; exit 2; }; }
for t in tar split gzip ssh scp stat df du; do need_bin "$t"; done

require_remote() {
    : "${DEST_HOST:?set DEST_HOST=hostname}"
    : "${DEST_DIR:?set DEST_DIR=/remote/target/path}"
    REMOTE="${DEST_USER}@${DEST_HOST}"
    SSH=(ssh -p "${SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
}

# -------------------------------------------------------------------- phase 1

phase1_labels() {
    banner "phase1/labels — tar+gzip"
    mkdir -p "${STAGE_DIR}"
    if [[ -f "${LABELS_ARCHIVE}" ]]; then
        echo "exists: ${LABELS_ARCHIVE}  ($(du -h "${LABELS_ARCHIVE}" | awk '{print $1}')) — delete to rebuild"
        return 0
    fi
    tar --no-xattrs --no-acls \
        -C "$(dirname "${LABELS_SRC}")" \
        -czf "${LABELS_ARCHIVE}.partial" \
        "$(basename "${LABELS_SRC}")"
    mv "${LABELS_ARCHIVE}.partial" "${LABELS_ARCHIVE}"
    du -h "${LABELS_ARCHIVE}"
}

phase1_images() {
    banner "phase1/images — plain tar (no compression on JPGs) + split ${CHUNK_SIZE}"
    mkdir -p "${IMAGES_PARTS_DIR}"

    # Marker so we know archive is complete and not a crashed build
    local done_marker="${IMAGES_PARTS_DIR}/.archive_complete"
    if [[ -f "${done_marker}" ]]; then
        echo "exists (complete): ${IMAGES_PARTS_DIR} — $(ls "${IMAGES_PARTS_DIR}" | grep -c '^part\.' || echo 0) chunks"
        echo "delete .archive_complete and any partial chunks to rebuild"
        return 0
    fi

    # Pre-flight space check
    local need have need_gb have_gb
    need=$(du -sb "${IMAGES_SRC}" | awk '{print $1}')
    have=$(df -B1 "${STAGE_DIR}" | tail -1 | awk '{print $4}')
    need_gb=$(( need / 1024**3 ))
    have_gb=$(( have / 1024**3 ))
    echo "images on source: ${need_gb} GiB   |   staging free: ${have_gb} GiB"
    if (( have < need + 10*1024**3 )); then
        echo "ERROR: need ~${need_gb}GB + 10GB buffer, staging has only ${have_gb}GB" >&2
        echo "Set STAGE_DIR=/path/with/more/space and retry" >&2
        return 3
    fi

    # Clear any partial state from a previous crashed archive
    rm -f "${IMAGES_PARTS_DIR}"/part.*

    tar --no-xattrs --no-acls \
        -C "$(dirname "${IMAGES_SRC}")" \
        -cf - "$(basename "${IMAGES_SRC}")" | \
      split -b "${CHUNK_SIZE}" -a 4 -d - "${IMAGES_PARTS_DIR}/part."

    touch "${done_marker}"
    echo "chunks created:"
    ls -sh1 "${IMAGES_PARTS_DIR}" | tail -5
    echo "total: $(ls "${IMAGES_PARTS_DIR}" | grep -c '^part\.' || echo 0) chunks"
}

# -------------------------------------------------------------------- phase 2

phase2_labels() {
    require_remote
    banner "phase2/labels — scp → ${REMOTE}:${DEST_DIR}/"
    [[ -f "${LABELS_ARCHIVE}" ]] || { echo "run 'archive' first" >&2; return 3; }
    "${SSH[@]}" "${REMOTE}" "mkdir -p '${DEST_DIR}'"
    local attempts=0
    until scp -P "${SSH_PORT}" "${LABELS_ARCHIVE}" "${REMOTE}:${DEST_DIR}/labels.tar.gz"; do
        ((++attempts))
        (( attempts > 5 )) && { echo "scp failed 5x" >&2; return 3; }
        echo "retry ${attempts}/5..."; sleep 5
    done
    echo "labels archive transferred."
}

phase2_images() {
    require_remote
    banner "phase2/images — scp chunks with per-chunk retry + skip-if-same-size"
    [[ -f "${IMAGES_PARTS_DIR}/.archive_complete" ]] || { echo "run 'archive' first" >&2; return 3; }
    "${SSH[@]}" "${REMOTE}" "mkdir -p '${DEST_DIR}/images_parts'"

    # Snapshot remote chunk sizes once; we'll skip chunks whose remote size matches local.
    local remote_sizes
    remote_sizes=$("${SSH[@]}" "${REMOTE}" \
        "ls -la '${DEST_DIR}/images_parts/' 2>/dev/null \
         | awk 'NR>1 && \$NF ~ /^part\\./ {print \$5\"|\"\$NF}'" || true)

    local total done_count=0 skipped=0
    total=$(ls "${IMAGES_PARTS_DIR}" | grep -c '^part\.' || echo 0)

    local f bn local_sz tag attempts
    for f in "${IMAGES_PARTS_DIR}"/part.*; do
        [[ -f "$f" ]] || continue
        ((++done_count))
        bn=$(basename "$f")
        local_sz=$(stat -c '%s' "$f")
        tag="[${done_count}/${total}] ${bn}"
        if grep -qx "${local_sz}|${bn}" <<<"${remote_sizes}"; then
            echo "${tag}: already on remote (size match), skip"
            ((++skipped))
            continue
        fi
        attempts=0
        until scp -P "${SSH_PORT}" "$f" "${REMOTE}:${DEST_DIR}/images_parts/${bn}.tmp" && \
              "${SSH[@]}" "${REMOTE}" "mv '${DEST_DIR}/images_parts/${bn}.tmp' '${DEST_DIR}/images_parts/${bn}'"; do
            ((++attempts))
            (( attempts > 5 )) && { echo "${tag}: failed 5x, aborting" >&2; return 3; }
            echo "${tag}: retry ${attempts}/5..."; sleep 5
        done
        echo "${tag}: ok"
    done
    echo "images transferred: ${done_count} chunks (${skipped} skipped as already-present)"
}

# -------------------------------------------------------------------- phase 3

phase3_labels() {
    require_remote
    banner "phase3/labels — extract on remote"
    "${SSH[@]}" "${REMOTE}" "set -e; cd '${DEST_DIR}'; tar -xzf labels.tar.gz; \
        echo '[remote] labels: '\$(ls -U labels_standalone 2>/dev/null | wc -l)' files'"
}

phase3_images() {
    require_remote
    banner "phase3/images — cat chunks | tar -x on remote"
    "${SSH[@]}" "${REMOTE}" "set -e; cd '${DEST_DIR}'; cat images_parts/part.* | tar -xf -; \
        echo '[remote] images dir size: '\$(du -sh Good2 2>/dev/null | awk '{print \$1}')"
}

# -------------------------------------------------------------------- dispatch

cleanup() {
    banner "cleanup — removing staged archives"
    rm -rf "${LABELS_ARCHIVE}" "${IMAGES_PARTS_DIR}"
    echo "cleaned ${STAGE_DIR} (only the two artefacts we created)"
}

usage() {
    cat <<EOF
Usage:  DEST_HOST=... DEST_DIR=... [DEST_USER=...] $0 STEP [WHAT]

  STEP: archive (1)  |  transfer (2)  |  extract (3)  |  cleanup  |  help
  WHAT: labels       |  images        |  all (default)

Env:   DEST_HOST DEST_USER DEST_DIR SSH_PORT STAGE_DIR CHUNK_SIZE WHAT

Typical flow:
  DEST_HOST=otherbox DEST_USER=pavan DEST_DIR=/data/laion_good2 $0 archive
  DEST_HOST=otherbox DEST_USER=pavan DEST_DIR=/data/laion_good2 $0 transfer
  DEST_HOST=otherbox DEST_USER=pavan DEST_DIR=/data/laion_good2 $0 extract
  $0 cleanup

Staging: needs ~1 TB free in STAGE_DIR (default /media/data_2/_laion_transfer_stage)
         for the images tar. Labels archive is ~100-200 MB.
EOF
}

STEP="${1:-help}"
case "${STEP}" in
    archive|1)
        case "${WHAT}" in labels) phase1_labels ;; images) phase1_images ;; all) phase1_labels; phase1_images ;; *) echo "bad WHAT"; exit 1 ;; esac ;;
    transfer|2)
        case "${WHAT}" in labels) phase2_labels ;; images) phase2_images ;; all) phase2_labels; phase2_images ;; *) echo "bad WHAT"; exit 1 ;; esac ;;
    extract|3)
        case "${WHAT}" in labels) phase3_labels ;; images) phase3_images ;; all) phase3_labels; phase3_images ;; *) echo "bad WHAT"; exit 1 ;; esac ;;
    cleanup) cleanup ;;
    help|-h|--help) usage ;;
    *) echo "unknown step: ${STEP}"; usage; exit 1 ;;
esac
