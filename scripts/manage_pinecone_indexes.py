import argparse
import logging
import sys

from pinecone import Pinecone
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manage_pinecone")


def list_indexes(pc):
    try:
        indexes = pc.list_indexes()
        # list_indexes may return list of names or objects with .name
        names = []
        for idx in indexes:
            try:
                names.append(idx.name)
            except Exception:
                names.append(str(idx))
        return names
    except Exception as e:
        logger.error(f"Failed to list Pinecone indexes: {e}")
        return []


def delete_index(pc, name):
    try:
        # Some clients expose delete_index or delete_index_by_name
        if hasattr(pc, "delete_index"):
            pc.delete_index(name)
        elif hasattr(pc, "Index"):
            # fallback: create Index object and call delete
            idx = pc.Index(name)
            if hasattr(idx, "delete_index"):
                idx.delete_index()
            else:
                # last resort: call pc._operation? Not supported
                raise RuntimeError("Delete operation not supported by this Pinecone client")
        else:
            raise RuntimeError("Delete operation not available")
        logger.info(f"Deleted index: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete index '{name}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="List or delete Pinecone indexes")
    parser.add_argument("--delete", help="Name of index to delete", default=None)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion (required when using --delete)",
    )
    args = parser.parse_args()

    if not settings.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set in settings/.env")
        sys.exit(1)

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    names = list_indexes(pc)
    if not names:
        logger.info("No indexes found.")
        return

    logger.info("Existing Pinecone indexes:")
    for n in names:
        logger.info(f" - {n}")

    if args.delete:
        if args.delete not in names:
            logger.error("Index '%s' not found", args.delete)
            return
        if not args.confirm:
            logger.warning(
                "Refusing to delete '%s' without --confirm. Re-run with --confirm to delete.",
                args.delete,
            )
            return
        success = delete_index(pc, args.delete)
        if success:
            logger.info("Deletion completed")
        else:
            logger.error("Deletion failed")


if __name__ == "__main__":
    main()
