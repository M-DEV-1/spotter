"""Stitch deck slide PNGs (+ optional episode clips) into a single MP4."""
import sys, glob
import numpy as np
import imageio.v2 as imageio

FPS = 30
SECONDS_TITLE = 4.0      # slides 1 & 9
SECONDS_TRACE = 7.0      # slide 5 (lots to read)
SECONDS_DEFAULT = 5.5
OUT = "outputs/spotter_submission.mp4"

slides = sorted(glob.glob("outputs/deck/slide_*.png"))
assert slides, "no slides found in outputs/deck/"

# Optional real episode clips to append (unsupervised fail → supervised recover)
clips = [c for c in [
    "outputs/episodes/classical_unsupervised.mp4",
    "outputs/episodes/classical_supervised.mp4",
] if __import__("os").path.exists(c)]

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])

def hold(img, secs):
    for _ in range(int(secs * FPS)):
        writer.append_data(img)

for i, path in enumerate(slides, 1):
    img = imageio.imread(path)[:, :, :3]
    if i in (1, len(slides)):
        secs = SECONDS_TITLE
    elif i == 5:
        secs = SECONDS_TRACE
    else:
        secs = SECONDS_DEFAULT
    hold(img, secs)
    print(f"  slide {i}: {secs}s")

# Append episode clips, resized to 1280x720, if any
for clip in clips:
    try:
        rdr = imageio.get_reader(clip)
        print(f"  appending {clip}")
        for frame in rdr:
            f = frame[:, :, :3]
            # letterbox/resize to 1280x720
            from PIL import Image
            im = Image.fromarray(f)
            im.thumbnail((1280, 720))
            canvas = Image.new("RGB", (1280, 720), (10, 14, 20))
            canvas.paste(im, ((1280 - im.width)//2, (720 - im.height)//2))
            writer.append_data(np.asarray(canvas))
        rdr.close()
    except Exception as e:
        print(f"  skip {clip}: {e}")

writer.close()
print(f"\nwrote {OUT}")
