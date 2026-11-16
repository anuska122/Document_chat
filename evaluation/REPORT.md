# Evaluation report

Measured locally on 2026-11-16 using Python 3.12.7, macOS ARM64, and tesseract 5.5.4.

## Dataset and comparison

10 development files represent two synthetic invoice identities in a pipe-table layout. The held-out set has 15 files: three different identities in a separate supplier-label/aligned-table layout, each rendered as native PDF, scanned PDF, mixed PDF, PNG and JPEG. These are five correlated renderings per identity, not 15 independent invoice sources. Mixed files contain a native cover page and a scanned invoice on page 2. All content is synthetic and authored for this project.

Baseline: original native-only PyMuPDF flattening, with images unsupported. The same new deterministic invoice parser is run on both text outputs to isolate extraction changes; invoice extraction did not exist in the original app. These clean synthetic layouts are deliberately limited and are not a supplier-diverse production benchmark.

## Held-out measurements

| Format (3 files each) | Pipeline | CER | WER | Header accuracy | Whole-row F1 | Page provenance | Failure rate | Mean extraction ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | baseline | 0.00% | 0.00% | 100.00% | 1.000 | 0.00% | 0.00% | 2.21 |
| native | improved | 0.00% | 0.00% | 100.00% | 1.000 | 100.00% | 0.00% | 2.05 |
| scanned | baseline | 100.00% | 100.00% | 0.00% | 0.000 | 0.00% | 100.00% | 0.48 |
| scanned | improved | 0.00% | 0.00% | 100.00% | 1.000 | 100.00% | 0.00% | 662.50 |
| mixed | baseline | 78.38% | 79.59% | 0.00% | 0.000 | 0.00% | 0.00% | 1.75 |
| mixed | improved | 0.00% | 0.00% | 100.00% | 1.000 | 100.00% | 0.00% | 651.70 |
| png | baseline | 100.00% | 100.00% | 0.00% | 0.000 | 0.00% | 100.00% | 0.03 |
| png | improved | 0.38% | 5.13% | 100.00% | 0.500 | 100.00% | 0.00% | 835.16 |
| jpeg | baseline | 100.00% | 100.00% | 0.00% | 0.000 | 0.00% | 100.00% | 0.07 |
| jpeg | improved | 0.38% | 5.13% | 100.00% | 0.500 | 100.00% | 0.00% | 802.19 |

All 15 improved extractions completed. All 120 labelled header values matched their references. The six image files had OCR row errors: line-item exact F1 was 0.500 for both PNG and JPEG, despite correct headers. Subtotal consistency warnings help expose incomplete rows; manual review remains necessary. Native-only mixed extraction returned the cover but missed the invoice, so its processing failure rate was zero while content accuracy was poor.

CER/WER normalize whitespace. Whole-row F1 requires all row fields to match, comparing numeric strings as Decimals. Page provenance checks document page position and reference-word overlap. It is not an assessment of whether an LLM claim is supported. Failures count as empty text, missing fields and missing citations; unsupported image formats count as baseline failures. Latency excludes queueing, chunking, embeddings and vector indexing.

## Integration and ingestion timing

25 focused regression tests passed, including actual Tesseract scans/mixed pages/images, corrupt/blank/encrypted inputs, size/page/time limits, metadata retrieval, rounding, duplicate review, correction/approval/version conflict/export, restart failure recovery, and no implicit Pinecone provisioning. Python compilation, JavaScript syntax, selected Flake8 error checks and `git diff --check` passed. Existing Pydantic class-Config and SQLAlchemy utcnow deprecation warnings remain.

An isolated smoke server passed actual upload/OCR/page-preview/original-file/review/approval/JSON/CSV checks. Its embedding, vector and chat adapters were deterministic substitutes; a citation link reached the expected page and an approved Decimal total returned 51.43. The browser visibly rendered the scanned invoice beside editable fields and source controls. Approval/export were independently verified through HTTP.

The API timing harness completed 10 development uploads with 0% failures against that smoke server. Mean upload-to-ready latency, including polling overhead, was 908.55 ms. This cannot be used as production ingestion latency because embedding/index services were substituted. Raw smoke measurements are in [smoke-ingestion-results.json](smoke-ingestion-results.json).

## Unmeasured and remaining limits

- No live Pinecone/LLM benchmark or Docker image build was run. Local Ollama had only an embedding model; no chat model was downloaded. The optional API harness supports real upload-to-ready timing and invoice-number answer/document/page checks using `--check-citations` once providers are ready.
- No observed human correction rate is available. Header mismatch proxy was zero on these references, but image row errors still need correction. `python -m evaluation.review_metrics` measures recorded review activity separately; automated smoke actions are not human study data.
- No authentication exists, so no multi-user isolation or authenticated audit-identity claim is made.
- Existing environment `pip check` failed because installed pdfplumber 0.11.7 requires pdfminer.six 20250506 while this repository pins 20221105. pdfplumber is not a declared project dependency; clean-environment installation was not verified. This conflict did not prevent the focused tests or extraction evaluation.
- The sample is small, English, clean, and synthetic. No estimates of business impact, real-world invoice accuracy, prompt-injection immunity or semantic LLM citation correctness are justified.

Reproduce with `python -m evaluation.run --split held_out`. Raw per-document results: [results.json](results.json).
