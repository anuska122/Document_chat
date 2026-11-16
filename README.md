# Document Intelligence and Q&A

FastAPI document chat extended with local OCR, page citations, typed invoice extraction, human review and JSON/CSV export. The browser UI is served by the existing backend at `/`. No separate frontend framework or build server is required.

## Start here: local app

The simplest setup runs entirely on your computer: **SQLite + Ollama**, with no Pinecone, PostgreSQL or Redis account required. The interface has three tabs: **Ask questions**, **Review invoice**, and **View document**.

1. Install Python 3.11+, [Ollama](https://ollama.com/download), and Tesseract (`brew install tesseract` on macOS, or `sudo apt-get install tesseract-ocr tesseract-ocr-eng` on Debian/Ubuntu).
2. From the repository root:

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   # Only for a new checkout; keep any existing .env:
   test -f .env || cp .env.example .env
   ```

3. Start Ollama (its desktop app or `ollama serve`), then explicitly download the models if needed:

   ```sh
   ollama pull all-minilm       # embeddings, about 45 MB
   ollama pull llama3.2:3b      # chat, about 2 GB
   python -m scripts.local_server
   ```

4. Open **http://127.0.0.1:8011**, add a PDF/image/text file, wait for **Ready**, and ask a question. Use **Review invoice** to correct and approve invoice data, or **View document** to inspect a cited page.

Keep `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.2:3b` in `.env` for this setup. The local server selects `all-minilm` embeddings and 384-dimensional local search automatically. Pinecone/Groq placeholders in `.env.example` are unused by this local configuration. Originals, page text, reviews and search vectors persist in `storage/local-preview`; this directory is ignored by Git. Back it up if you want to retain uploaded documents.

The app is **single-user and local-only**. Model downloads are explicit; the server does not download a model or substitute a fixed answer when chat is unavailable. Small local models can be slow and can misinterpret text; quote checks verify provenance, not every claim's meaning.

For the existing PostgreSQL/Pinecone deployment path, follow [Local setup](#local-setup) below and run `main:app` on port 8000.

## Current verification

- **36 regression tests passed**, including actual local Tesseract OCR, mixed PDFs, page provenance, invoice validation/review/export, and chat quote/citation validation.
- Real local Q&A checks passed for a synthetic invoice and the exact summary question `what is the summary`, including checked source excerpts and working page navigation. See [chat integration checks](evaluation/CHAT_CHECK.md).
- The held-out extraction evaluation completed **15 synthetic files**, with **120/120 header values matching**. PNG/JPEG line-item F1 was **0.50**; manual review remains necessary. These correlated synthetic examples are not a production accuracy benchmark. See [evaluation results and limitations](evaluation/REPORT.md).
- Live Pinecone integration, a Docker image build and a clean dependency installation remain unverified. The existing environment has a documented pdfplumber/pdfminer dependency conflict; it did not prevent the focused checks.

## Architecture and data flow

1. `POST /api/ingest` streams a bounded upload, stores its original bytes under a generated ID, computes SHA-256, creates a PostgreSQL processing record, and returns **202** with a status URL.
2. One bounded background queue processes one document at a time. PyMuPDF parsing and Tesseract run in a separate process group, which is killed on timeout. PostgreSQL records queued → extracting → indexing → ready, or failed.
3. Each PDF page uses native text if it has at least 30 alphanumeric characters and no replacement characters. Pages with large image coverage (>40%) use OCR to capture scanned regions even when a native header exists. OCR **replaces** native text for that page. Original numbering is retained, including blank pages. Images become one-page documents.
4. Page text, method, normalized bounding boxes `[left, top, right, bottom]`, real Tesseract word scores (when returned), and warnings are saved in an additive `intelligence_documents` JSON table. Existing `documents` metadata remains intact. Local originals live in `STORAGE_DIR`; back up both storage and PostgreSQL.
5. Each page passes separately through the existing semantic/fixed chunker and configured embeddings. Pinecone receives the complete chunk text, document ID, filename, page number and extraction method. Retrieval is restricted to ready, page-aware document IDs, including the caller's explicit selection. Failed jobs cannot be retrieved even if vector cleanup fails.
6. The existing Ollama/Groq provider returns typed answer points with numeric source IDs and short exact supporting quotes. Ollama uses schema-constrained JSON output. The server verifies quotes against supplied text and builds citation markers itself; a misnumbered quote can be resolved only to an unambiguous source page. Unknown IDs, fabricated quotes and ambiguous matches are rejected. Invalid output is retried once and then shown as an answer-validation error, rather than falsely claiming the documents lack information. Evidence is placed in a separate untrusted user payload; no tools are exposed. Summary evidence uses page order, with tables of contents after substantive pages. Source buttons open the original page preview. Native line regions are highlighted only when an exact, unique line matches available evidence; OCR and uncertain geometry use page navigation. Similarity is not presented as answer confidence.
7. A separate deterministic invoice parser extracts explicitly labelled fields and simple line tables. Pydantic validates the typed schema; machine extraction is immutable. Review writes corrected values, validation warnings, version, timestamp, local actor, action and a snapshot to the audit history. Approval with unresolved warnings requires a review note. Editing an approved record returns it to needs review.
8. JSON exports contain machine and reviewed values, status and audit. CSV exports include raw/normalized fields, evidence and review status, and escape formula-leading cells. Approved totals are summed using `Decimal`, separately by currency, through `/api/invoices/summary` rather than LLM arithmetic.

The original checkout had no frontend, authentication, ownership model, queue or tests. **This remains a local single-user application.** All documents and chat history are shared by anyone who can reach the server. Bind to loopback; do not expose it as a multi-user service without adding verified authentication and ownership checks to every document, preview, review, export, analytics, vector and history path. The local actor label is not an authenticated identity. No authentication-isolation claim is made.

## Local setup

Use Python 3.11 (the Docker target; local validation also ran on Python 3.12):

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Do not overwrite an existing `.env`; merge the example's settings instead. The repository's existing `sentence-transformers`, Groq/OpenAI and Pinecone dependencies/providers are reused.

Install open-source Tesseract and English language data:

```sh
# macOS: Homebrew installs Tesseract and bundled English data
brew install tesseract
# Debian/Ubuntu (already included in Dockerfile)
sudo apt-get install tesseract-ocr tesseract-ocr-eng

tesseract --version
tesseract --list-langs
```

Tesseract was chosen because it runs locally on CPU, fits the existing Python/Docker environment, and supplies TSV words, coordinates and engine-specific confidence scores. These scores are **not calibrated probabilities**. No OCR web service, Python OCR wrapper, or runtime model download is used. Installation downloads the engine and bundled `eng.traineddata`; other languages require installing the corresponding Tesseract language package and setting `OCR_LANGUAGE`. See [official Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [TSV usage](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).

Start the existing infrastructure and configured LLM:

```sh
docker compose up -d postgres redis
# Install/start Ollama separately, then explicitly download the configured chat model:
ollama pull llama3.2:3b
```

**Model downloads:** The local setup uses `llama3.2:3b` (about 2 GB), configured with `LLM_MODEL=llama3.2:3b`. The original `llama3.1` model is about 5 GB and was too resource-intensive on the 8 GB development Mac. Default HF embeddings download `sentence-transformers/all-mpnet-base-v2` on first embedding use unless cached; prefetch explicitly with:

```sh
python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-mpnet-base-v2")'
```

Alternatively reuse the existing Ollama embedding provider, with an explicitly downloaded model and matching Pinecone/embedding dimensions. `all-minilm` uses 384 dimensions; the default HF model uses 768. Existing OpenAI/Groq providers require their own keys and may incur normal provider charges. No new service is required. Pinecone must refer to an **existing** matching index; automatic index creation has been disabled to prevent accidental provisioning.

```sh
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000` (API docs at `/docs`). Use **one ASGI worker and one application instance**. Queue state is in memory; job status and results are persistent. Restarted pending jobs become failed with a re-upload message rather than remaining stuck. This is deliberately a small local queue, not a distributed durable worker. Redis history degrades gracefully when unavailable; PostgreSQL and Pinecone initialization are required for the normal app.

Docker includes OCR dependencies. Keep `.env`, local originals and virtual environments outside the build context (`.dockerignore`). Supply environment variables at runtime, connect the app to your PostgreSQL/Redis/Ollama hosts, and mount persistent storage at the configured `STORAGE_DIR`. Do not run multiple replicas with this queue design. No deployment was performed.

### Environment variables

| Setting | Default / requirement |
| --- | --- |
| `DATABASE_URL` | PostgreSQL URL; `asyncpg` is now included |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Existing account/index |
| `PINECONE_DIMENSION`, `EMBEDDING_DIMENSION` | Must match each other and the index/model; default 768 |
| `EMBEDDING_PROVIDER` | Existing `hf`, `ollama`, or `openai` |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` |
| `OLLAMA_BASE_URL`, `LLM_MODEL` | `http://localhost:11434`, configured `LLM_MODEL` (local example: `llama3.2:3b`) |
| `LLM_PROVIDER` | Existing `ollama` or `groq` |
| `OLLAMA_CONTEXT_WINDOW` | 4096 tokens; retrieved evidence is bounded to leave room for instructions and output |
| `OLLAMA_MAX_OUTPUT_TOKENS` | 512; also bounded by `LLM_MAX_TOKENS` |
| `GROQ_API_KEY` | Required by existing Settings; use a non-secret placeholder with Ollama |
| `OPENAI_API_KEY` | Only needed for OpenAI embeddings |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` | Existing Redis settings |
| `STORAGE_DIR` | `storage`; originals must survive restarts |
| `MAX_FILE_SIZE_MB` | 10 |
| `MAX_DOCUMENT_PAGES` | 50 |
| `PROCESSING_TIMEOUT_SECONDS` | 180 for extraction; overall job bounded at this + 120 seconds |
| `OCR_PAGE_TIMEOUT_SECONDS` | 30 |
| `OCR_LANGUAGE` | `eng`; corresponding traineddata must be installed |

## Formats, validation and limits

PDF (text, scanned or mixed), PNG, JPEG and existing UTF-8 TXT are supported. Magic bytes must match the extension for PDF/images. File size is checked while reading. Empty uploads and unsupported extensions fail immediately; corrupt/encrypted files, excessive pages, unreadable pages, missing OCR dependencies and timeouts produce actionable processing failures. Raster rendering is capped at 25 million pixels at 200 DPI. Blank pages retain their number; a wholly blank document fails. The queue accepts 20 waiting jobs plus the active job.

Invoice headers require explicit labels such as `Vendor:`, `Supplier Name:`, `Invoice Number:`, `Invoice No.:`, `Invoice Date:`, `Due Date:`, `Currency:`, `Subtotal:`, `Tax:`/`VAT:` and `Grand Total:`. Simple line rows use `description | quantity | unit price | amount`, or description followed by three whitespace-separated numbers. Other layouts remain editable and generate missing/incomplete-table warnings. Repeated contradictory labels are ambiguous. Dates are normalized only when ISO YYYY-MM-DD; ambiguous regional dates remain unresolved. Currency symbols alone are ambiguous. Money accepts unambiguous dot decimals, optionally comma-grouped thousands. No dates, currencies, vendors or amounts are guessed. LLMs are not used for structured extraction.

Arithmetic uses `Decimal`, ROUND_HALF_UP at the currency's supported minor unit, and tolerance of **one minor unit** (USD/NPR/EUR etc. 0.01, JPY/KRW 1, KWD/BHD/OMR/JOD/TND 0.001). Unknown currency precision is explicitly unvalidated. Discounts, shipping, credits, withholding and rounding adjustments disable summary arithmetic checks; individual line mismatches remain review warnings, not automatic rejection. Table completeness cannot be proven. SHA-256 duplicates and normalized vendor + invoice-number duplicates are warnings only; files are retained. Approved totals include intentional duplicates and disclose their document IDs; reviewers must consider duplicate warnings before using the totals.

## Tests and evaluation

```sh
python -m pytest -q
node --check web/app.js
python -m evaluation.run --split held_out
python -m evaluation.run --split development --output evaluation/development-results.json
```

Tests use isolated SQLite via `aiosqlite`, synthetic documents, real local Tesseract, and substituted vector/embedding/chat adapters. They never contact configured hosted providers or modify the application's PostgreSQL data. OCR tests skip explicitly if Tesseract is missing. Coverage includes native/scanned/mixed/images, bad/blank/encrypted files, limits/timeouts, page metadata, decimal validation, ambiguous fields, duplicate warnings, background failures, restart recovery, corrections, approval, version conflicts, exports and citation filtering. There is no authentication in the app to test user isolation against.

`evaluation/generate.py` creates 10 development files (two synthetic identities, one layout family) and 15 held-out files (three different identities, a different label/table/placement layout), five formats each. Files within each group are correlated renderings, **not independent invoice sources**. No private records are used. Generated binaries are ignored; generator, labels and measured output are reproducible. The small clean-English dataset does not establish performance on real suppliers, handwriting, camera distortion, complex tables or other languages.

The baseline reproduces original PyMuPDF native-only flattened extraction and unsupported images. The same new invoice parser is applied to both baseline and improved text to isolate ingestion effects; the original app did not have invoice extraction. CER/WER use normalized whitespace and edit distance. Header accuracy is exact normalized-value match. Line-item F1 requires an entire row to match (numbers compared as Decimals). Page-provenance accuracy checks page index and reference-word overlap; it does **not** measure generated-answer citation support. The report includes failures rather than excluding them from accuracy. Latency is extraction only, excluding queue/embeddings/indexing.

Measured results and limitations: [evaluation/REPORT.md](evaluation/REPORT.md), raw [evaluation/results.json](evaluation/results.json).

For full upload-to-ready latency (uploads are retained, and configured provider usage applies):

```sh
python -m evaluation.ingestion --base-url http://127.0.0.1:8000 --split held_out --check-citations
# Observed correction rates after real human reviews; null until reviews exist:
python -m evaluation.review_metrics
```

`--check-citations` additionally asks for the invoice number and measures exact answer content and gold document/page citation precision. This is a narrow grounded-answer check, not a general semantic citation benchmark.

Reference mismatches are a **correction proxy**, not measured human correction activity. The review metric separately reports observed corrected documents/header fields. No human correction-rate or business-impact claim is made from synthetic extraction results.

For a fully local application without Pinecone/PostgreSQL, run:

```sh
# Explicit model downloads, only if not already installed:
ollama pull all-minilm
ollama pull llama3.2:3b
python -m scripts.local_server
# Open http://127.0.0.1:8011
```

This server uses persistent SQLite and originals under `storage/local-preview`, real Ollama `all-minilm` embeddings (384 dimensions), local cosine retrieval, and the **configured chat provider/model** from `.env`. Set `LOCAL_DATA_DIR` to choose another local directory. The normal `main:app` entry point still uses PostgreSQL/Pinecone. Local indexes are rebuilt from preserved page text if missing; original extraction and review history remain intact. The local vector store is intended for a small single-user workspace, not a large collection or multiple replicas.

`python -m scripts.smoke_server` is now a compatibility alias for this real local application. The former fixed-response test server has been removed. Test substitutions belong only in regression tests. `/api/chat/status` checks whether the configured local chat model is installed and reachable; unavailable chat shows a setup message, disables submission, and never displays an answer or citation. A provider failure returns HTTP 503 and a separate error message. Chat availability does not block document preview or invoice review.

## Demo walkthrough

The workspace shows one task at a time: **Ask questions**, **Review invoice**, or **View document**. Source links open the document tab; invoice edits stay in place while switching tabs. Original extraction, review history, and page text are expandable details.

1. Upload `evaluation/fixtures/development/0_scanned.pdf` after running the generator. Watch processing finish and open the page preview/original.
2. Ask “What is the invoice number?” in the normal app. Click the returned source to navigate to its document/page. Insufficient or invalidly cited answers are refused.
3. Check the invoice fields and source buttons. Correct a value, save, and inspect the original machine output and review history.
4. Approve the invoice. If warnings or duplicates exist, document your review in the note first. Export JSON and CSV.
5. Upload the same file again to see an exact duplicate warning. Upload a differently encoded file with the same vendor/number for a possible business duplicate.
6. Use **Approved invoice totals** for exact Decimal sums by currency. Generic cross-invoice comparisons and natural-language analytics are deferred; chat arithmetic requests are directed to this structured-data workflow.

## Known limitations and compatibility

- No existing authentication or ownership controls; single-user only. CORS is limited to the default local UI origins. A public/multi-user deployment requires authentication and per-user filtering across all paths.
- One local queue; no automatic retries, retention policy, horizontal scaling, resumable jobs or malware scanning. Failed jobs retain originals for diagnosis/re-upload. Back up and manage storage explicitly.
- Existing legacy uploads have no stored originals/pages. They remain in existing SQL/vector storage but are excluded from new cited retrieval; re-upload originals to use OCR/page review. No unverifiable page numbers are invented.
- `POST /api/ingest` now returns 202 and a polling URL instead of synchronous final token/chunk counts. Final counts are on the document record. Existing clients must poll; the bundled UI does so.
- Chat history endpoints remain available, but prior turns are not fed into answer generation to avoid stale evidence across document selections. Ask self-contained questions.
- Prompt separation, verified source quotes and lack of tools reduce document-instruction risk; they do not prove semantic grounding or eliminate prompt injection. Live LLM citation correctness has not been benchmarked.
- Conservative invoice formats and date normalization; complex invoice structures need manual correction. OCR geometry is stored but highlights use reliable native lines only. OCR text is searchable and pages are navigable.
- Pinecone integration and Docker image build remain unverified. Real local Ollama answers were checked separately: an invoice-number answer and a document summary returned selected-document citations with working page previews. These are integration checks, not a semantic citation benchmark. See [local chat checks](evaluation/CHAT_CHECK.md).
