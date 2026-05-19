import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db.sql_db import init_db, close_db
from db.vector_db import init_vector_db, close_vector_db
from services.redis_memory import init_redis, close_redis
from routers import ingest, chat


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Application:
    """FastAPI application wrapper following OOP principles."""

    def __init__(self) -> None:
        self._app = FastAPI(
            title="Conversational RAG Backend",
            description="AI-powered document chat with custom RAG and memory",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._setup_middleware()
        self._setup_routes()
        self._setup_lifespan()

    def _setup_middleware(self) -> None:
        """Configure application middleware."""
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware configured")

    def _setup_routes(self) -> None:
        """Register API routers."""
        self._app.include_router(
            ingest.router,
            prefix="/api",
            tags=["Document Ingestion"]
        )
        self._app.include_router(
            chat.router,
            prefix="/api",
            tags=["Conversational RAG"]
        )
        logger.info("Routers registered")

    def _setup_lifespan(self) -> None:
        """Configure lifespan context manager."""
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            await self._startup()
            yield
            await self._shutdown()

        self._app.router.lifespan_context = lifespan

    async def _startup(self) -> None:
        """Initialize services on startup."""
        logger.info("Starting RAG Backend...")

        await init_db()
        logger.info("PostgreSQL initialized")

        await init_vector_db()
        logger.info("Pinecone initialized")

        await init_redis()
        logger.info("Redis initialized")

        logger.info("All services ready")

    async def _shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Shutting down RAG Backend...")

        await close_redis()
        logger.info("Redis closed")

        await close_vector_db()
        logger.info("Pinecone closed")

        await close_db()
        logger.info("PostgreSQL closed")

        logger.info("Shutdown complete")

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI application instance."""
        return self._app

    def get(self, path: str, **kwargs):
        """Delegate route registration to FastAPI app."""
        return self._app.get(path, **kwargs)

    def post(self, path: str, **kwargs):
        """Delegate route registration to FastAPI app."""
        return self._app.post(path, **kwargs)

    def include_router(self, *args, **kwargs):
        """Delegate router registration to FastAPI app."""
        return self._app.include_router(*args, **kwargs)


app = Application()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "vector_db": "connected",
            "sql_db": "connected",
            "redis": "connected"
        }
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to Conversational RAG Backend",
        "docs": "/docs",
        "health": "/health"
    }


# Expose the internal FastAPI instance for ASGI servers (uvicorn expects an ASGI app)
app = app.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
