#!/usr/bin/env bash
set -e

mkdir -p data/race_highlights/train
mkdir -p data/race_highlights/test/grandprix
mkdir -p data/race_highlights/test/qualifying

download() {
    local target="$1"
    local url="$2"

    if [ ! -f "$target" ]; then
        yt-dlp --remote-components ejs:github -S "vcodec:h264:vp9,res" --merge-output-format mp4 -o "$target" "$url"
    fi

    echo "Ready: $target"
}

download "data/race_highlights/train/2025_qatar_gp.mp4" "https://www.youtube.com/watch?v=tkVp7dcBd6c"
download "data/race_highlights/train/2025_spanish_gp.mp4" "https://www.youtube.com/watch?v=I9fphFbTVz8"
download "data/race_highlights/train/2025_german_gp_sprint.mp4" "https://www.youtube.com/watch?v=k16usQXnvhE"
download "data/race_highlights/train/2025_italian_gp.mp4" "https://www.youtube.com/watch?v=R82xflovqx8"

download "data/race_highlights/test/grandprix/2025_australian_gp_sprint.mp4" "https://www.youtube.com/watch?v=_4PfoIJJKh4"
download "data/race_highlights/test/grandprix/2025_british_gp.mp4" "https://www.youtube.com/watch?v=eHcKnGRiwCg"

download "data/race_highlights/test/qualifying/2025_portuguese_q2.mp4" "https://www.youtube.com/watch?v=xeyfn2inQpg"
download "data/race_highlights/test/qualifying/2025_spanish_q2.mp4" "https://www.youtube.com/watch?v=avXXzvnlOBk"
