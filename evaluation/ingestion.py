"""Opt-in complete API ingestion timing, including queued processing and indexing.
This uploads synthetic files to the selected server and retains them there.
Configured provider usage may incur that provider's normal charges.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
import httpx
from evaluation.generate import build


async def run(base_url, split, check_citations=False):
    results = []
    async with httpx.AsyncClient(base_url=base_url, timeout=180) as client:
        for item in [r for r in build() if r["split"] == split]:
            started = time.perf_counter()
            path = Path(__file__).parent / item["path"]
            try:
                with path.open("rb") as source:
                    uploaded = await client.post(
                        "/api/ingest", files={"file": (path.name, source)}
                    )
                uploaded.raise_for_status()
                document_id = uploaded.json()["document_id"]
                while time.perf_counter() - started < 360:
                    response = await client.get("/api/documents/" + document_id)
                    response.raise_for_status()
                    record = response.json()
                    if record["status"] in {"ready", "failed"}:
                        break
                    await asyncio.sleep(0.25)
                else:
                    raise TimeoutError("Document did not complete within 360 seconds")
                results.append(
                    dict(
                        path=item["path"],
                        kind=item["kind"],
                        document_id=document_id,
                        status=record["status"],
                        message=record["message"],
                        ingestion_latency_ms=round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                    )
                )
                if check_citations and record["status"] == "ready":
                    try:
                        answer = await client.post(
                            "/api/chat",
                            json=dict(
                                session_id="evaluation-" + document_id,
                                message="What is the invoice number?",
                                document_ids=[document_id],
                            ),
                        )
                        answer.raise_for_status()
                        body = answer.json()
                        sources = body.get("sources", [])
                        expected_page = len(item["reference_pages"])
                        correct = sum(
                            source.get("document_id") == document_id
                            and source.get("page_number") == expected_page
                            for source in sources
                        )
                        results[-1]["citation_doc_page_precision"] = (
                            correct / len(sources) if sources else 0
                        )
                        results[-1]["has_correct_citation"] = bool(correct)
                        results[-1]["invoice_number_answer_correct"] = item["fields"][
                            "invoice_number"
                        ] in body.get("response", "")
                    except Exception as exc:
                        results[-1]["citation_check_error"] = str(exc)
            except Exception as exc:
                results.append(
                    dict(
                        path=item["path"],
                        kind=item["kind"],
                        status="harness_error",
                        message=str(exc),
                    )
                )
    return dict(
        server=base_url,
        split=split,
        documents=len(results),
        failure_rate=sum(r["status"] != "ready" for r in results) / len(results),
        note="Includes polling overhead. Inspect server provider configuration before interpreting or comparing results.",
        results=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--split", choices=["development", "held_out"], default="held_out")
    p.add_argument(
        "--check-citations",
        action="store_true",
        help="Also call chat and check invoice-number answers against gold document/page labels",
    )
    p.add_argument("--output", default="evaluation/ingestion-results.json")
    args = p.parse_args()
    result = asyncio.run(run(args.base_url, args.split, args.check_citations))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
