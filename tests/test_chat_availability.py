import io
import json
import urllib.error
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from services import rag, local_vectors
from db.sql_db import save_intelligence, intelligence_record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "installed,available",
    [
        ([], False),
        ([{"name": "llama3.1:latest", "capabilities": ["completion"]}], True),
        ([{"name": "all-minilm:latest", "capabilities": ["embedding"]}], False),
    ],
)
async def test_chat_readiness_checks_real_model_name(monkeypatch, installed, available):
    pipeline = rag.CustomRAGPipeline()
    monkeypatch.setattr(
        rag.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps({"models": installed}).encode()),
    )
    status = await pipeline.chat_status()
    assert status["available"] is available
    assert bool(status["message"]) is not available


@pytest.mark.asyncio
async def test_chat_service_unavailable_has_no_generated_answer(monkeypatch):
    from routers import chat

    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    monkeypatch.setattr(
        chat,
        "run_rag",
        AsyncMock(
            side_effect=RuntimeError(
                "Answer generation is unavailable. Check the configured chat service and try again."
            )
        ),
    )
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={
                "session_id": "test",
                "message": "Summarize the selected document",
                "document_ids": ["doc_a"],
            },
        )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
    assert "response" not in response.json() and "sources" not in response.json()


@pytest.mark.asyncio
async def test_local_vector_search_persists_and_respects_selection(
    database, monkeypatch
):
    monkeypatch.setattr(local_vectors.settings, "EMBEDDING_DIMENSION", 2)
    for doc in ("a", "b", "failed"):
        await save_intelligence(
            dict(document_id=doc, status="failed" if doc == "failed" else "ready")
        )
        await local_vectors.upsert_embeddings(
            doc,
            ["Source evidence " + doc],
            [[1.0, 0.0]],
            [dict(source=doc, page_number=2)],
        )
    assert (await intelligence_record("a"))["search_index"][0]["metadata"][
        "page_number"
    ] == 2
    result = await local_vectors.query_vectors([1.0, 0.0], document_ids=["a", "failed"])
    assert len(result) == 1 and result[0]["metadata"]["document_id"] == "a"
    assert result[0]["score"] == 1
    assert await local_vectors.query_vectors([1.0, 0.0], document_ids=[]) == []
    with pytest.raises(ValueError):
        await local_vectors.upsert_embeddings("a", ["text"], [[1.0]], [{}])


@pytest.mark.asyncio
async def test_offline_chat_status_is_actionable(monkeypatch):
    pipeline = rag.CustomRAGPipeline()

    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(rag.urllib.request, "urlopen", offline)
    status = await pipeline.chat_status()
    assert (
        status["available"] is False
        and "Start the local chat service" in status["message"]
    )


def test_context_budget_preserves_only_evidence_sent_to_model(monkeypatch):
    pipeline = rag.CustomRAGPipeline()
    monkeypatch.setattr(rag.settings, "MAX_CONTEXT_LENGTH", 30)
    original = {
        "metadata": {
            "text": "invoice details " * 100,
            "document_id": "a",
            "page_number": 3,
        },
        "score": 0.8,
    }
    selected = pipeline.limit_context([original, original])
    assert len(selected) == 1
    assert len(rag.chunker.tokenizer.encode(selected[0]["metadata"]["text"])) <= 30
    assert selected[0]["metadata"]["page_number"] == 3
    assert original["metadata"]["text"] == "invoice details " * 100


@pytest.mark.asyncio
async def test_grounded_answers_build_citations_from_verified_quotes(monkeypatch):
    pipeline = rag.CustomRAGPipeline()
    context = pipeline.build_context(
        [
            {
                "metadata": {
                    "text": "Payroll setup uses salary profiles. Invoice number: DEV-001.",
                    "page_number": 2,
                    "source": "Report.pdf",
                }
            }
        ]
    )
    messages = await pipeline.build_messages("what is the summary", context, "test")
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "Payroll setup uses salary profiles.",
                    "source": 1,
                    "quote": "Payroll setup uses salary profiles.",
                }
            ]
        }
    )
    monkeypatch.setattr(pipeline, "_generate_raw_response", AsyncMock(return_value=raw))
    assert (
        await pipeline.generate_response(messages)
        == "Payroll setup uses salary profiles. [Source 1]"
    )
    for invalid in [
        {
            "claims": [
                {"text": "Unsupported claim", "source": 99, "quote": "Payroll setup"}
            ]
        },
        {
            "claims": [
                {
                    "text": "Unsupported claim",
                    "source": 1,
                    "quote": "A fabricated quote",
                }
            ]
        },
        {
            "claims": [
                {"text": "Claim [Source N]", "source": 1, "quote": "Payroll setup"}
            ]
        },
    ]:
        pipeline._generate_raw_response.return_value = json.dumps(invalid)
        with pytest.raises(RuntimeError, match="could not be checked"):
            await pipeline.generate_response(messages)


@pytest.mark.asyncio
async def test_formatting_failure_is_retried_without_claiming_missing_documents(
    monkeypatch,
):
    pipeline = rag.CustomRAGPipeline()
    context = pipeline.build_context(
        [
            {
                "metadata": {
                    "text": "Payroll configuration is documented.",
                    "page_number": 1,
                    "source": "Report.pdf",
                }
            }
        ]
    )
    messages = await pipeline.build_messages("what is the summary", context, "test")
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "Payroll configuration is documented.",
                    "source": 1,
                    "quote": "Payroll configuration is documented.",
                }
            ]
        }
    )
    generation = AsyncMock(side_effect=["Here is a summary [Source N]", raw])
    monkeypatch.setattr(pipeline, "_generate_raw_response", generation)
    answer = await pipeline.generate_response(messages)
    assert generation.await_count == 2 and "[Source 1]" in answer
    assert "not contain" not in answer


@pytest.mark.asyncio
async def test_quote_can_resolve_a_valid_but_misnumbered_source(monkeypatch):
    pipeline = rag.CustomRAGPipeline()
    chunks = [
        {
            "metadata": {
                "document_id": "a",
                "text": "Appendix API endpoints.",
                "source": "Report.pdf",
                "page_number": 11,
            }
        },
        {
            "metadata": {
                "document_id": "a",
                "text": "Payroll setup uses salary profiles.",
                "source": "Report.pdf",
                "page_number": 2,
            }
        },
    ]
    messages = await pipeline.build_messages(
        "what is the summary", pipeline.build_context(chunks), "test"
    )
    raw = json.dumps(
        {
            "claims": [
                {
                    "text": "Payroll setup uses salary profiles.",
                    "source": 1,
                    "quote": "Payroll setup uses salary profiles.",
                }
            ]
        }
    )
    quotes = {}
    assert pipeline.render_grounded_answer(raw, messages, quotes).endswith("[Source 2]")
    assert quotes == {2: "Payroll setup uses salary profiles."}
    chunks.append(
        {
            "metadata": {
                "document_id": "b",
                "text": "Payroll setup uses salary profiles.",
                "source": "Other.pdf",
                "page_number": 4,
            }
        }
    )
    messages = await pipeline.build_messages(
        "summary", pipeline.build_context(chunks), "test"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        pipeline.render_grounded_answer(raw, messages)


@pytest.mark.asyncio
async def test_summary_prioritizes_reading_order_over_appendix_similarity(monkeypatch):
    pipeline = rag.CustomRAGPipeline()
    chunks = [
        {
            "score": 0.9,
            "metadata": {
                "document_id": "a",
                "source": "Report.pdf",
                "text": "Appendix API endpoints.",
                "page_number": 11,
            },
        },
        {
            "score": 0.4,
            "metadata": {
                "document_id": "a",
                "source": "Report.pdf",
                "text": "Overview: payroll setup uses salary profiles.",
                "page_number": 2,
            },
        },
        {
            "score": 0.8,
            "metadata": {
                "document_id": "a",
                "source": "Report.pdf",
                "text": "Contents........ Overview........ Appendix........",
                "page_number": 1,
            },
        },
    ]
    monkeypatch.setattr(
        pipeline, "retrieve_relevant_chunks", AsyncMock(return_value=chunks)
    )
    generation = AsyncMock(
        return_value="Payroll setup uses salary profiles. [Source 1]"
    )
    monkeypatch.setattr(pipeline, "generate_response", generation)
    monkeypatch.setattr(rag, "add_to_history", AsyncMock())
    result = await pipeline.process_query("what is the summary", "test", ["a"])
    assert result["sources"][0]["page_number"] == 2
    evidence = json.loads(generation.call_args.args[0][1]["content"])[
        "untrusted_evidence"
    ]
    assert [item["page"] for item in evidence] == [2, 11, 1]
