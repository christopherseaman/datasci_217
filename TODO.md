# TODO

- [ ] **Lecture 07: lists over paragraphs** for the Data-Ink Ratio and Lie Factor explanations, still prose.
- [ ] **Revisit Lecture 07's distribution of visualization content by package** (instructor, 2026-10-07): which topics belong to matplotlib, pandas .plot, seaborn, and Altair, and whether each demo's core route covers its package's block.
- [ ] **Humor candidates to place** (instructor, 2026-10-07):
    - People surnamed Null whose names forms reject or treat as empty (Christopher Null); another sentinel one-liner.
    - xkcd 2109, "Invisible Formatting" (hidden meaning in data).
    - xkcd 534, "Genetic Algorithms" (maximize vs minimize; fits Lecture 10 model training).
    - xkcd 435, "Purity" (fields arranged by purity), already saved as `11/media/xkcd_435.png` and shown nowhere.
    - "No yak shaving!" image: https://cdn.dribbble.com/userupload/29322615/file/original-2fa98d729c6c94ed63416c2787194bd5.png
- [ ] **Rebuild the 06 midterm generator to the 05 standard** (instructor, 2026-10-14). The audit of `06/mid-term/generate_data.py` found:
    - Clinical values drawn independently: baseline A1c, SBP, age, and weight correlate near 0; gender drives nothing.
    - SBP change is a flat uniform draw unrelated to baseline; weight change is unrelated to A1c change.
    - Clip spikes: 75 week-12 SBP rows at exactly 90, 203 baseline weights at 48 kg.
    - Rechecks are independent of the first reading (correlation -0.15, mean gap 20.7 mmHg).
    - Ages uniform from 30 to 86; visit times flat 08:00 to 17:00; week-12 interval uniform 81 to 87 days.
    - Decide: E04 visits routed into `visits_south.csv`, and planted SBP 70 and 250 boundary rows.
    - A rebuild changes every 06 expected value, manifest hash, and grading constant.
