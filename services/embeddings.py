import asyncio
import json
import logging
import random
import urllib.error
import urllib.request
from typing import List, Sequence

import tiktoken

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER.lower()
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = None
        self.hf_model = None
        self._hf_model_lock = asyncio.Lock()
        self._openai_module = None

        if self.provider == "hf":
            self.model = settings.HF_EMBEDDING_MODEL
            logger.info(
                f"Hugging Face embedding model configured: {self.model}"
            )
        elif self.provider == "ollama":
            self.model = settings.OLLAMA_EMBEDDING_MODEL
            self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
            self.ollama_max_tokens = settings.OLLAMA_EMBEDDING_MAX_TOKENS
            self.ollama_tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info(
                f"Ollama embedding model configured: {self.model} "
                f"(dimension: {self.dimension})"
            )
        elif self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
                )
            import openai
            from openai import AsyncOpenAI

            self._openai_module = openai
            logger.info(
                "Initializing OpenAI embedding model: %s",
                settings.EMBEDDING_MODEL,
            )
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(
                f"OpenAI embedding model ready (dimension: {self.dimension})"
            )
        else:
            raise ValueError(
                "Unsupported EMBEDDING_PROVIDER="
                f"{settings.EMBEDDING_PROVIDER!r}. "
                "Use 'openai', 'hf', or 'ollama'."
            )

    async def generate_embedding(self, text: str) -> List[float]:
        """Generation of embedding for single text"""
        try:
            text = self._clean_text(text)
            if not text:
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.dimension

            if self.provider == "hf":
                model = await self._get_hf_model()
                embedding = await asyncio.to_thread(
                    model.encode, text, convert_to_numpy=False
                )
                logger.debug(
                    f"Generated HF embedding of length {len(embedding)}"
                )
                return self._validate_embedding(embedding)

            if self.provider == "ollama":
                embeddings = await asyncio.to_thread(
                    self._ollama_embeddings_sync, [text]
                )
                logger.debug(
                    "Generated Ollama embedding length %s",
                    len(embeddings[0]),
                )
                return embeddings[0]

            # Retry on rate limits with exponential backoff for OpenAI
            max_attempts = 5
            base_delay = 1.0
            params = self._openai_params(text)
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await self.client.embeddings.create(**params)
                    embedding = response.data[0].embedding
                    logger.debug(
                        f"Generated embedding of dimension {len(embedding)}"
                    )
                    return self._validate_embedding(embedding)
                except self._openai_module.RateLimitError as rl_err:
                    if attempt == max_attempts:
                        logger.error(
                            "Rate limit reached (attempt %s/%s): %s",
                            attempt,
                            max_attempts,
                            rl_err,
                        )
                        raise
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    logger.warning(
                        "Rate limited generating embedding; retrying in "
                        "%.1fs (attempt %s/%s)",
                        delay,
                        attempt,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    # re-raise non-rate-limit exceptions immediately
                    raise

        except Exception as e:
            logger.error(
                f"Error generating embedding: {str(e)}", exc_info=True
            )
            raise

    async def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """Generates embeddings for multiple texts in batches"""
        try:
            cleaned_texts = [self._clean_text(text) for text in texts]
            all_embeddings = []

            total_batches = (
                len(cleaned_texts) + batch_size - 1
            ) // batch_size

            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                logger.info(
                    "Processing batch %s/%s",
                    i // batch_size + 1,
                    total_batches,
                )

                non_empty_indexes = [
                    idx for idx, text in enumerate(batch) if text
                ]
                batch_embeddings = [[0.0] * self.dimension for _ in batch]
                non_empty_texts = [batch[idx] for idx in non_empty_indexes]

                if not non_empty_texts:
                    all_embeddings.extend(batch_embeddings)
                    continue

                if self.provider == "hf":
                    model = await self._get_hf_model()
                    embeddings = await asyncio.to_thread(
                        model.encode, non_empty_texts, convert_to_numpy=False
                    )
                    for idx, embedding in zip(non_empty_indexes, embeddings):
                        batch_embeddings[idx] = self._validate_embedding(
                            embedding
                        )
                    all_embeddings.extend(batch_embeddings)
                    continue

                if self.provider == "ollama":
                    embeddings = await asyncio.to_thread(
                        self._ollama_embeddings_sync, non_empty_texts
                    )
                    for idx, embedding in zip(non_empty_indexes, embeddings):
                        batch_embeddings[idx] = embedding
                    all_embeddings.extend(batch_embeddings)
                    continue

                # Retry loop for each batch for OpenAI
                max_attempts = 5
                base_delay = 1.0
                params = self._openai_params(non_empty_texts)
                for attempt in range(1, max_attempts + 1):
                    try:
                        response = await self.client.embeddings.create(
                            **params
                        )
                        embeddings = [
                            self._validate_embedding(item.embedding)
                            for item in response.data
                        ]
                        for idx, embedding in zip(
                            non_empty_indexes, embeddings
                        ):
                            batch_embeddings[idx] = embedding
                        all_embeddings.extend(batch_embeddings)
                        break
                    except self._openai_module.RateLimitError as rl_err:
                        if attempt == max_attempts:
                            logger.error(
                                "Rate limit reached for batch "
                                "(attempt %s/%s): %s",
                                attempt,
                                max_attempts,
                                rl_err,
                            )
                            raise
                        delay = base_delay * (
                            2 ** (attempt - 1)
                        ) + random.uniform(0, 0.5)
                        logger.warning(
                            "Rate limited generating batch embeddings; "
                            "retrying in %.1fs (attempt %s/%s)",
                            delay,
                            attempt,
                            max_attempts,
                        )
                        await asyncio.sleep(delay)
                    except Exception:
                        # re-raise non-rate-limit exceptions immediately
                        raise

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    async def _get_hf_model(self):
        if self.hf_model is not None:
            return self.hf_model

        async with self._hf_model_lock:
            if self.hf_model is None:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    f"Loading Hugging Face embedding model: {self.model}"
                )
                self.hf_model = await asyncio.to_thread(
                    SentenceTransformer, self.model
                )
                logger.info("Hugging Face embedding model loaded")

        return self.hf_model

    def _clean_text(self, text: str) -> str:
        return (text or "").replace("\n", " ").strip()

    def _openai_params(self, input_text):
        params = {
            "input": input_text,
            "model": self.model,
        }
        if self.model.startswith("text-embedding-3"):
            params["dimensions"] = self.dimension
        return params

    def _validate_embedding(self, embedding: Sequence[float]) -> List[float]:
        values = [float(value) for value in embedding]
        if len(values) != self.dimension:
            raise ValueError(
                f"{self.provider} embedding dimension {len(values)} "
                "does not match "
                f"EMBEDDING_DIMENSION={self.dimension}."
            )
        return values

    def _ollama_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self._post_ollama(
                "/api/embed",
                {
                    "model": self.model,
                    "input": texts,
                },
            )
            embeddings = response.get("embeddings")
            if embeddings is not None:
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Ollama returned {len(embeddings)} embeddings "
                        f"for {len(texts)} inputs"
                    )
                return [
                    self._validate_embedding(embedding)
                    for embedding in embeddings
                ]
        except urllib.error.HTTPError as exc:
            error = self._ollama_http_error(exc)
            if exc.code in {404, 405}:
                pass
            elif self._is_ollama_context_error(exc.code, error):
                logger.warning(
                    "Ollama batch input exceeded context length; "
                    "embedding texts in smaller pieces."
                )
                return [
                    self._embed_ollama_text_with_splitting(text)
                    for text in texts
                ]
            else:
                raise error from exc

        embeddings = []
        for text in texts:
            try:
                response = self._post_ollama(
                    "/api/embeddings",
                    {
                        "model": self.model,
                        "prompt": text,
                    },
                )
            except urllib.error.HTTPError as exc:
                error = self._ollama_http_error(exc)
                if self._is_ollama_context_error(exc.code, error):
                    embeddings.append(
                        self._embed_ollama_text_with_splitting(text)
                    )
                    continue
                raise error from exc

            embedding = response.get("embedding")
            if embedding is None:
                raise RuntimeError(
                    "Ollama embedding response did not include an embedding"
                )
            embeddings.append(self._validate_embedding(embedding))

        return embeddings

    def _embed_ollama_text_with_splitting(self, text: str) -> List[float]:
        pieces = self._split_for_ollama(text)
        if not pieces:
            return [0.0] * self.dimension

        piece_embeddings = []
        for piece in pieces:
            try:
                response = self._post_ollama(
                    "/api/embed",
                    {
                        "model": self.model,
                        "input": [piece],
                    },
                )
                embeddings = response.get("embeddings")
                if not embeddings:
                    raise RuntimeError(
                        "Ollama embedding response did not include embeddings"
                    )
                piece_embeddings.append(
                    self._validate_embedding(embeddings[0])
                )
            except urllib.error.HTTPError as exc:
                error = self._ollama_http_error(exc)
                if (
                    self._is_ollama_context_error(exc.code, error)
                    and len(piece) < len(text)
                ):
                    nested_embeddings = [
                        self._embed_ollama_text_with_splitting(nested_piece)
                        for nested_piece in self._split_for_ollama(
                            piece,
                            max_tokens=max(32, self.ollama_max_tokens // 2),
                        )
                    ]
                    piece_embeddings.extend(nested_embeddings)
                    continue
                raise error from exc

        return self._average_embeddings(piece_embeddings)

    def _split_for_ollama(
        self,
        text: str,
        max_tokens: int | None = None,
    ) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []

        token_limit = max_tokens or self.ollama_max_tokens
        tokens = self.ollama_tokenizer.encode(text)
        if len(tokens) <= token_limit:
            return [text]

        pieces = []
        for start in range(0, len(tokens), token_limit):
            piece = self.ollama_tokenizer.decode(
                tokens[start:start + token_limit]
            ).strip()
            if piece:
                pieces.append(piece)
        return pieces

    def _average_embeddings(
        self,
        embeddings: List[List[float]],
    ) -> List[float]:
        if not embeddings:
            return [0.0] * self.dimension

        averaged = [
            sum(embedding[i] for embedding in embeddings) / len(embeddings)
            for i in range(self.dimension)
        ]
        return self._validate_embedding(averaged)

    def _post_ollama(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.ollama_base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.ollama_base_url}. "
                "Make sure Ollama is running."
            ) from exc

    def _ollama_http_error(self, exc: urllib.error.HTTPError) -> RuntimeError:
        body = exc.read().decode("utf-8", errors="replace")
        message = body.strip() or exc.reason
        return RuntimeError(
            f"Ollama embedding request failed with HTTP {exc.code}: "
            f"{message}"
        )

    def _is_ollama_context_error(
        self,
        status_code: int,
        error: RuntimeError,
    ) -> bool:
        message = str(error).lower()
        return (
            status_code == 400
            and "input length" in message
            and "context length" in message
        )


embedding_generator = EmbeddingGenerator()


async def generate_embedding(text: str) -> List[float]:
    """Wrapper for single embedding generation"""
    return await embedding_generator.generate_embedding(text)


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Wrapper for batch embedding generation"""
    return await embedding_generator.generate_embeddings_batch(texts)
