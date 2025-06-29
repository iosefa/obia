cd training_crowns_raw
mkdir -p train/images train/masks val/images val/masks
shopt -s nullglob
for img in images/*.png; do
    base=${img##*/}; base=${base%.png}
    msk=masks/${base}_mask.png
    if (( RANDOM % 5 == 0 )); then
        mv "$img" val/images/
        mv "$msk" val/masks/
    else
        mv "$img" train/images/
        mv "$msk" train/masks/
    fi
done
rmdir images masks 2>/dev/null || true
echo "train  $(ls train/images | wc -l)"
echo "val    $(ls val/images | wc -l)"
cd ..