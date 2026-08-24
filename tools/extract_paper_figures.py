#!/usr/bin/env python3
"""Extract the paper figures used by the static project page.

The crop boxes are expressed in PDF points and intentionally include the
figure caption. Re-run this script after replacing the source preprint.
"""

from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "408_Towards_a_Universal_Sequen.pdf"
OUTPUT = ROOT / "imgs" / "paper"

# page number: [(file stem, (x0, top, x1, bottom)), ...]
CROPS = {
    5: [("figure-03-method-ranks", (45, 238, 566, 480))],
    8: [("figure-07-unsupervised-selection", (45, 70, 566, 208))],
    18: [("figure-10-optuna", (45, 70, 566, 395))],
    20: [("figure-11-random-baseline", (45, 70, 566, 395))],
    23: [
        ("figure-13-aggregation-by-method", (45, 70, 566, 214)),
        ("figure-14-aggregation-map", (45, 245, 566, 445)),
    ],
    30: [("figure-15-unsupervised-trajectories", (45, 70, 566, 363))],
}


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Missing source PDF: {SOURCE}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(SOURCE) as paper:
        for page_number, crops in CROPS.items():
            page = paper.pages[page_number - 1]
            for stem, bounds in crops:
                image = page.crop(bounds).to_image(resolution=220, antialias=True)
                destination = OUTPUT / f"{stem}.png"
                image.save(destination)
                print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
