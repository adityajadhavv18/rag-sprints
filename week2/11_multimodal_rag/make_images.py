"""Phase 1: draw 6 synthetic HVMC images whose key fact lives ONLY in the picture.

Each image adds a detail the text corpus never states (and never contradicts):
cap colours match the lab catalog, unit names match the directory.
Writes images/*.png and images/manifest.jsonl (the hidden fact is for scoring, never for retrieval).

uv run week2/11_multimodal_rag/make_images.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
OUT = HERE / "images"
FOOTER = "SYNTHETIC IMAGE: fictional HVMC hospital, created for retrieval testing. Not medical guidance."


def canvas(title: str, bg: str = "white"):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor(bg)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis("off")
    ax.text(50, 71, title, ha="center", va="center", fontsize=15, weight="bold",
            color="white" if bg != "white" else "black")
    fig.text(0.5, 0.015, FOOTER, ha="center", fontsize=7, color="gray")
    return fig, ax


def save(fig, name: str):
    fig.savefig(OUT / name, facecolor=fig.get_facecolor())
    plt.close(fig)


def order_of_draw():
    """Tubes in the order they are drawn. Colours are the six cap colours the lab catalog uses."""
    fig, ax = canvas("HVMC Phlebotomy: Order of Draw")
    tubes = [("light blue", "#8ecae6"), ("gold", "#e9c46a"), ("mint green", "#95d5b2"),
             ("lavender", "#c8b6e2"), ("pink", "#f4acb7"), ("gray", "#adb5bd")]
    for i, (name, colour) in enumerate(tubes):
        x = 10 + i * 15
        ax.add_patch(Rectangle((x - 3, 12), 6, 34, facecolor="#f1f3f5", edgecolor="black"))  # tube body
        ax.add_patch(Rectangle((x - 4, 46), 8, 9, facecolor=colour, edgecolor="black"))     # cap
        ax.text(x, 59, str(i + 1), ha="center", fontsize=18, weight="bold")
        ax.text(x, 7, name, ha="center", fontsize=10)
    ax.annotate("", xy=(92, 66), xytext=(8, 66), arrowprops=dict(arrowstyle="->", lw=1.5))
    save(fig, "img01_order_of_draw.png")


def iv220_panel():
    """Front panel layout: the silence (bell) button sits top-left of the keypad, STOP is bottom-right."""
    fig, ax = canvas("Kestrel IV-220: Front Panel")
    ax.add_patch(FancyBboxPatch((15, 5), 70, 58, boxstyle="round,pad=1", facecolor="#dee2e6", edgecolor="black"))
    ax.add_patch(Rectangle((20, 62), 60, 3, facecolor="#e63946", edgecolor="black"))       # alarm light bar
    ax.add_patch(Rectangle((22, 42), 56, 15, facecolor="#1d3557", edgecolor="black"))      # display
    ax.text(50, 49.5, "125 mL/h", ha="center", va="center", fontsize=20, color="white", family="monospace")
    keys = [["SILENCE", "▲", "MENU"], ["◀", "OK", "▶"], ["START", "▼", "STOP"]]
    colours = {"SILENCE": "#ffb703", "START": "#2a9d8f", "STOP": "#e63946"}
    for r, row in enumerate(keys):
        for c, label in enumerate(row):
            x, y = 30 + c * 20, 33 - r * 11
            ax.add_patch(FancyBboxPatch((x - 7, y - 4), 14, 8, boxstyle="round,pad=0.5",
                                        facecolor=colours.get(label, "white"), edgecolor="black"))
            ax.text(x, y, label, ha="center", va="center", fontsize=10, weight="bold")
    save(fig, "img02_iv220_front_panel.png")


def vm8_screen():
    """Screen layout and colours: HR green, SpO2 cyan, NIBP white, RR yellow."""
    fig, ax = canvas("Talon VM-8: Main Screen", bg="black")
    xs = [i / 4 for i in range(0, 200)]
    ys = [52 + (6 if i % 25 == 10 else -2 if i % 25 == 12 else 0) for i in range(0, 200)]
    ax.plot([5 + x for x in xs], ys, color="#38b000", lw=1.5)                   # ECG trace, top-left
    panels = [(78, 52, "HR", "72", "#38b000"), (78, 30, "SpO2 %", "97", "#00e5ff"),
              (25, 15, "NIBP mmHg", "118/76", "white"), (78, 10, "RR", "16", "#ffd60a")]
    for x, y, label, value, colour in panels:
        ax.text(x, y + 9, label, ha="center", fontsize=11, color=colour)
        ax.text(x, y, value, ha="center", va="center", fontsize=30, weight="bold", color=colour)
    save(fig, "img03_vm8_screen.png")


def hand_hygiene_chart():
    """Bar heights carry the fact: 3 South is lowest and the only unit under the 90% target."""
    units = ["2 North", "2 South", "3 North", "3 South", "4 North", "4 South", "5 North"]
    values = [94, 92, 96, 71, 91, 93, 95]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.bar(units, values, color="#4c6ef5")
    ax.axhline(90, color="#e63946", ls="--", lw=1.5)
    ax.text(3, 91.5, "target 90%", ha="center", color="#e63946", fontsize=10)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Compliance (%)")
    ax.set_title("HVMC Hand Hygiene Compliance by Unit, Q3 2026", weight="bold")
    fig.text(0.5, 0.015, FOOTER, ha="center", fontsize=7, color="gray")
    save(fig, "img04_hand_hygiene_chart.png")


def wristband_key():
    """Colour key: red = allergy, yellow = fall risk, purple = DNR, green = latex allergy."""
    fig, ax = canvas("HVMC Patient Wristband Colour Key")
    bands = [("#e63946", "Allergy"), ("#ffd60a", "Fall risk"), ("#7b2cbf", "Do not resuscitate (DNR)"),
             ("#2a9d8f", "Latex allergy")]
    for i, (colour, meaning) in enumerate(bands):
        y = 55 - i * 14
        ax.add_patch(FancyBboxPatch((10, y - 3), 30, 6, boxstyle="round,pad=1", facecolor=colour, edgecolor="black"))
        ax.add_patch(Circle((36, y), 1.2, facecolor="white", edgecolor="black"))  # snap
        ax.text(47, y, meaning, va="center", fontsize=14)
    save(fig, "img05_wristband_key.png")


def vm8_night_mode_card():
    """Text-only card: the answer is printed words, which CLIP is known to read poorly."""
    fig, ax = canvas("Talon VM-8 Quick Card: Night Mode")
    ax.add_patch(Rectangle((8, 8), 84, 56, facecolor="#fff8e1", edgecolor="black"))
    lines = ["1. Press MENU.", "2. Select Display, then Night.", "3. The screen dims to 30% brightness.",
             "4. Alarm volume does NOT change.", "5. Touch the screen to return to day mode."]
    for i, line in enumerate(lines):
        ax.text(13, 56 - i * 10, line, fontsize=14, family="monospace", va="center")
    save(fig, "img06_vm8_night_mode_card.png")


MANIFEST = [
    {"id": "img01", "file": "img01_order_of_draw.png", "title": "Phlebotomy order of draw",
     "fact": "Light blue is drawn first, then gold, mint green, lavender, pink, gray."},
    {"id": "img02", "file": "img02_iv220_front_panel.png", "title": "Kestrel IV-220 front panel",
     "fact": "The SILENCE button is top-left of the keypad; STOP is bottom-right; START bottom-left."},
    {"id": "img03", "file": "img03_vm8_screen.png", "title": "Talon VM-8 main screen",
     "fact": "HR is green, SpO2 is cyan, NIBP is white, RR is yellow."},
    {"id": "img04", "file": "img04_hand_hygiene_chart.png", "title": "Hand hygiene compliance by unit, Q3 2026",
     "fact": "3 South is lowest (71%) and the only unit below the 90% target."},
    {"id": "img05", "file": "img05_wristband_key.png", "title": "Patient wristband colour key",
     "fact": "Red = allergy, yellow = fall risk, purple = DNR, green = latex allergy."},
    {"id": "img06", "file": "img06_vm8_night_mode_card.png", "title": "Talon VM-8 night mode quick card",
     "fact": "MENU > Display > Night; screen dims to 30%; alarm volume does not change."},
]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for draw in (order_of_draw, iv220_panel, vm8_screen, hand_hygiene_chart, wristband_key, vm8_night_mode_card):
        draw()
    with open(OUT / "manifest.jsonl", "w") as f:
        for row in MANIFEST:
            f.write(json.dumps(row) + "\n")
    for row in MANIFEST:
        print(f"{row['id']}  {row['file']:32s} {row['fact']}")
