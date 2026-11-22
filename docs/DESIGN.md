# Design direction

## 1. Concept
A claims desk tool, not a showcase. The liquidator's question is always the same: *can I trust what was
read, and how much time do I have?* Every screen answers those two things first: the route (ready / to
review, with the reasons) and the art. 148 term, then the form itself with what each agent read on it.

## 2. Motif — "registration marks"
The first agent finds the four corners of the printed frame before anything else is read. The same four
corner marks frame the cards that carry a verdict (the proof on the home page, the case verdict, the
generator). On the form viewer the motif becomes literal: a frame drawn on every field that was read,
coloured by who can act on it.

## 3. Mood
Quiet, exact, slightly technical: the paper of an insurance form (the faint 20 px grid of section 13 at the
top of every page), monospaced identifiers (case ids, plates, field ids), no decoration that does not carry
information.

## 4. Principles
- Colour: OKLCH graphite scale (hue 230-240, near-zero chroma) and **one** accent, petrol
  (`oklch(0.47 0.09 205)`). Three state colours, each with one meaning: amber = a person must look,
  red = inconsistent, green = confirmed by a person. White background, no beige.
- Type: Geist for text, Geist Mono for identifiers and labels (both OFL, self-hosted: the CSP allows no
  third-party origin).
- Confidence is never a percentage in a paragraph: a five-tick meter whose colour says what to do.
- Charts and bars are SVG with sizes as attributes — the Content-Security-Policy forbids inline styles.

## 5. Consistency
Same chip for the same state everywhere (queue, case, home). Same order on every case: verdict → term →
checks → truth comparison (synthetic only) → fields → steps. Italian in the product, English in the code.

## 6. Craft detail
- The overlay is an SVG in the form's own coordinate system (1240 × 1754), so frames stay on their fields at
  any width, and every frame is a link to its row in the field table.
- Empty boxes and empty text fields are not framed: the overlay shows what was read, not the template.
- Dark theme re-derives every token instead of inverting; the form image stays on white (it is paper).
