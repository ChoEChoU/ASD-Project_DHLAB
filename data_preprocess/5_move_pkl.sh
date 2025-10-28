# 원본 루트 디렉토리
src_root="./data/1017_unmatching_videos"

# 대상 루트 디렉토리 (여기에 같은 구조로 이동됨)
dst_root="./data/1017_unmatching_features"

rsync -av --info=progress2,stats --partial --remove-source-files \
  --include='*.pkl' --include='*/' --exclude='*' \
  "$src_root/" "$dst_root/"

echo "🎉 모든 .pkl 파일 이동 완료!"