# Manuscript Extension Plan

This is a short note about how to extend this IVC manuscript into a fuller paper, 
without breaking anything that already works.

---

## One Important Rule

- **Do not change the top‑level structure.**
  - Keep the same section headings and numbering (Introduction, Related Work,
    Methods, Results, Discussion, Conclusion, etc.).
  - Keep the existing figures and tables, labels, and cross‑references.
  - When you extend, do it *inside* the current sections or in clearly marked
    appendices—no renaming or changing the order of major sections.

If you follow this rule, the artifact bundle, references, and cross‑section logic
will all remain valid.

---

## How to Extend It

Safe ways to expand without breaking anything:

- Inside each section you can:
  - Add short subsections (e.g., “4.2.1 Training Details”) under the existing
    headings.
  - Enrich paragraphs with extra intuition, examples, or brief related‑work
    contrasts.
  - Point to specific files in `artifacts/` (JSON metrics, scripts, figures) when
    more detail would reassure a careful reader.
- You can also add small, focused tables or figures that refine existing claims
  (for instance, a per‑severity breakdown of corruption results) as long as you
  preserve the existing numbering and references.

---

## Places That Naturally Need More Detail

These are good candidates for expansion, but none are mandatory:

- **Complexity Grid / MDP pipeline** – more intuition, maybe a small diagram or an
  example walk‑through of one Scale × Corruption cell.
- **Methods** – clearer descriptions of model variants, training schedules, and
  prompt policies, especially when you can ground them in the provided scripts.
- **Results + Discussion** – more narrative around:
  - How performance shifts across scales,
  - The ROI crossover between accuracy and latency,
  - What the attention maps and spatial entropy are really telling us.
- **Limitations / Future Work** – a more candid discussion of domain gaps and how
  the same evaluation philosophy could be reused on other datasets or models.

---

## Staying Aligned With the Artifacts

As you extend the paper:

- Whenever you add a numerical claim, try to base it on something that already
  lives in `artifacts/` (JSON metrics, figures, or scripts) so the extended
  version remains reproducible.
- If you run new experiments, clearly mark them as **new** and, ideally, drop any
  supporting data or scripts next to the existing ones so a future reader can tell
  what belongs to the IVC submission and what was added later.

---
