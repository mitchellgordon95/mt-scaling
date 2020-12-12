visible=(${CUDA_VISIBLE_DEVICES//,/ })
# If expr returns 0, bash quits when we're using set -e
maxid=$(expr "${#visible[@]}" - 1) || true
visible_mapping=$(seq -s ' ' 0 $maxid)
echo "--device-ids $visible_mapping"
