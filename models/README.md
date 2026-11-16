# Application schemas

This directory contains typed request, response and extraction schemas. For setup and the complete workflow, see the [main README](../README.md).

| File | Purpose |
| --- | --- |
| `schemas.py` | Chat requests/responses, source-page references, chunking strategies and the legacy ingestion response schema. |
| `answers.py` | `GroundedAnswer` and `CitedClaim`: concise answer points with a numeric source ID and supporting quote. |
| `invoice.py` | Invoice values, page/text evidence, line items and versioned review requests. |
| `db_models.py` | Legacy re-export of `Document` and `Base`; the SQLAlchemy definitions live in [`db/sql_db.py`](../db/sql_db.py). |

## Answers and citations

The model returns typed claims rather than formatting citation links itself. The server checks source IDs and supporting quotes against the supplied document text, then constructs page citations. A misnumbered quote can resolve to another supplied source only when its page is unambiguous. A matching quote verifies provenance; it does not prove that every paraphrased claim is semantically correct.

## Invoice extraction and review

Each `Value` preserves its original `raw` text, optional `normalized` value, status (`present`, `missing` or `ambiguous`) and page/text evidence. The invoice schema contains vendor name, invoice number, invoice/due dates, currency, subtotal, tax, grand total and typed line items.

Monetary values are serialized as strings and validated using `Decimal`. Dates use ISO format when unambiguous. Missing values remain explicit rather than being guessed. Arithmetic checks and duplicate warnings are implemented separately in [`services/invoices.py`](../services/invoices.py).

`ReviewRequest` includes the corrected invoice, action (`save` or `approve`), expected version and review note. The server retains original machine extraction separately from reviewed values and records actions in the document audit history. The app currently has no authentication; its local actor label is not an authenticated user identity.

## Persistence and compatibility

The additive `intelligence_documents` table stores processing status, extraction provenance, original/reviewed invoice values and audit history as JSON. Existing document metadata remains in `documents`. Originals and local search data are managed outside this directory; model weights and uploaded files must not be added here or committed.

`IngestionResponse` remains for compatibility with earlier code, but the current upload route returns HTTP 202 with a document ID and polling URL. Final processing results are retrieved from `/api/documents/{document_id}`.
