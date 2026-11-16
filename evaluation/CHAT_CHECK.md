# Local chat integration check — 2026-11-16

The UI test server that returned a fixed response has been removed. Both local entry points (`scripts.local_server` and compatibility alias `scripts.smoke_server`) now use real embeddings, persistent local cosine search and the configured chat provider. PostgreSQL/Pinecone remain available through the normal `main:app` entry point.

On the 8 GB development Mac, the original llama3.1 model timed out on the first document question. With user approval, it was removed and replaced with llama3.2:3b. The local `.env` was updated; no hosted provider was added. Context defaults to 4096 tokens, output is capped at 512 tokens, and retrieved text is bounded before generation. Evidence is supplied as JSON data, with numeric citation IDs. No fixed response or citation is substituted if generation fails.

Actual HTTP checks, with no substituted embedding/vector/chat adapter:

| Check | Observed outcome |
| --- | --- |
| Synthetic invoice: “What is the invoice number?” | HTTP 200; DEV-001 matched the uploaded invoice; citation linked its page 1. About 32.4 seconds including retrieval and generation. |
| User-provided contribution report: three-point summary | HTTP 200; three generated bullets with citations restricted to the selected document. All returned page-preview URLs returned HTTP 200 with image content. About 60.4 seconds. |
| Model missing / provider failure | Availability notice or HTTP 503; UI shows a separate error, with no answer or citations. |
| Preserved uploads | Both originals, page text, corrections, review status and audit history survived migration from temporary to persistent local storage. |

These two live answers verify integration, numeric citation references and page navigation. They do not establish general answer accuracy, semantic support for every summary claim, or production latency. The extraction evaluation remains separate. Private document text and generated summary content are intentionally excluded from this report.

32 regression tests passed. JavaScript syntax, selected Flake8 error checks, and `git diff --check` passed. The earlier extraction measurements in `REPORT.md` were not rerun or changed by this chat fix.

Start with `python -m scripts.local_server`, open `http://127.0.0.1:8011`, select a processed document and ask a question. `/api/chat/status` checks service/model availability; installed-model availability alone cannot guarantee a successful inference request. Local generation can take tens of seconds on this hardware.

## Exact-question regression — summary validation

The question `what is the summary` exposed a remaining format/attribution failure. A prose citation failure had incorrectly been reported as absent document evidence. Generation now returns a typed `claims` object with text, numeric source ID and a short supporting quote; local Ollama output is constrained by that schema. Quotes are checked against supplied text before the server builds source markers. A valid but misnumbered quote may resolve to another supplied source only when its page is unambiguous. Unknown IDs, invented quotes and ambiguous source-page matches remain rejected. One format retry is allowed; exhausted validation produces a distinct error, not a missing-document claim. The source preview displays the checked excerpt.

For overview questions, retrieved evidence is ordered by page, with table-of-contents material after substantive content, before applying the context limit. This summarizes retrieved pages rather than proving complete coverage of every page.

The exact question passed twice against the real local model: HTTP 200, a substantive summary, a page-2 citation, excerpt text verified against that page, and a working page-preview URL. Observed end-to-end times were about 40.5 and 7.5 seconds. These two observations are not a latency benchmark or proof of semantic correctness for arbitrary questions.

36 tests passed after this change, including invalid-format retry, fabricated quote rejection, unknown source rejection, ambiguous quote rejection, unique source resolution, and summary evidence ordering. Existing tests were preserved.
