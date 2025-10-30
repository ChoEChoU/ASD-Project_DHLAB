#!/usr/bin/env bash
set -Eeuo pipefail

# 🔧 원본 루트 디렉토리 (비디오 프레임이 저장된 경로)
src_root="./data/preprocessed/unmatching_videos_frames"

# 🔧 대상 루트 디렉토리 (.pkl feature 파일이 저장될 경로)
dst_root="./data/preprocessed/unmatching_features"

# 📦 .pkl 파일만 동일한 폴더 구조로 이동
rsync -av --info=progress2,stats --partial --remove-source-files \
  --include='*.pkl' --include='*/' --exclude='*' \
  "$src_root/" "$dst_root/"

echo "🎉 모든 .pkl 파일 이동 완료!"