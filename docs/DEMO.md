# Five-minute walkthrough

1. **Panoramica** — the claim of the product in one line, and the numbers measured on the holdout set
   (forms never used for tuning, harder photos, two unseen handwriting styles).
2. **Coda** — the liquidator's queue, ordered by the art. 148 term. Filter by reason: uncertain boxes,
   signatures to check, poor photo, inconsistencies, injuries. «Automatiche» are cases whose data need no
   re-checking.
3. Open a case marked *da rivedere*:
   - left, the form straightened by the registrar, with a frame on every field that was read (petrol = read
     with confidence, amber = look at it, red = inconsistent, green = confirmed by a person); each frame links
     to its row;
   - right, the verdict with its reasons, the term for the offer (30 / 60 / 90 days, *provvisorio* when an
     uncertain signature or injuries box could change it), the checks, and — synthetic data only — what was
     really written.
4. Sign in (`liquidatore` / `liquidatore-demo-2026`) and correct one flagged field, e.g. an uncertain
   signature: the term and the route are recomputed at once, the reading is kept in an append-only log.
5. **Genera un caso** — choose a story (crossroads, rear-end...), defects (missing signature of B, a cross
   scratched out, a wrong codice fiscale...) and the capture quality. «Foto mossa e scura» shows the quality
   gate: the system says it cannot trust the marks instead of reading them wrong.
6. **Qualità** — dev vs holdout, defects found, the robustness curve, and the comparison with sending the
   whole page to the vision model.
7. **Processo** — where the time goes and why cases end up in the queue, with what would remove each reason.
