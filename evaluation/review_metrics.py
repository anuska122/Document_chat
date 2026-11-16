"""Observed human correction metrics from persisted local review events."""
import asyncio
import json
from db.sql_db import intelligence_records, close_db


async def main():
    reviewed = 0
    corrected = 0
    fields = 0
    changed = 0
    for record in await intelligence_records():
        if not record.get("audit") or not record.get("original"):
            continue
        reviewed += 1
        machine = record["original"]
        human = record["reviewed"]
        differences = sum(
            machine["fields"][k].get("normalized") != v.get("normalized")
            for k, v in human["fields"].items()
        )
        fields += len(machine["fields"])
        changed += differences
        corrected += bool(differences or machine["line_items"] != human["line_items"])
    print(
        json.dumps(
            dict(
                reviewed_documents=reviewed,
                document_correction_rate=corrected / reviewed if reviewed else None,
                header_field_correction_rate=changed / fields if fields else None,
            )
        )
    )
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
