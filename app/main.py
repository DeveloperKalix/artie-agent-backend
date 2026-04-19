import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from starlette.responses import HTMLResponse, JSONResponse
from tinyfish import TinyFish  # Modern 2026 SDK import

from app.backend.core.scheduler import build_scheduler
from app.routes.conversations import router as conversations_router
from app.routes.exchanges import router as exchanges_router
from app.routes.news import router as news_router
from app.routes.plaid import router as plaid_router
from app.routes.profile import router as profile_router
from app.routes.recommendations import router as recommendations_router
from app.routes.skills import router as skills_router
from app.routes.snaptrade import router as snaptrade_router

# Surface INFO logs from our own modules under uvicorn's default stream handler.
# ``force=True`` ensures this takes effect even if uvicorn has already attached
# a root handler in a reload-spawned worker.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop background infrastructure alongside the ASGI app.

    Wires the scheduler, registers the hourly news-ingest job, and kicks off
    a boot-time ingest so the news table is populated before the first user
    request hits ``/news`` or ``/recommendations``.
    """
    from app.backend.services.news import get_news_agent

    scheduler = build_scheduler()
    news_agent = get_news_agent()

    interval_min = int(os.getenv("NEWS_INGEST_INTERVAL_MINUTES", "60"))
    scheduler.add_job(
        news_agent.ingest,
        trigger="interval",
        minutes=interval_min,
        id="news_ingest",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    app.state.news_agent = news_agent
    logger.info(
        "[lifespan] scheduler started, news_ingest every %d minutes",
        interval_min,
    )

    # Boot-time ingest (fire-and-forget so startup isn't blocked).
    asyncio.create_task(_run_boot_ingest(news_agent))

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("[lifespan] scheduler stopped")


async def _run_boot_ingest(news_agent) -> None:
    try:
        count = await news_agent.ingest()
        logger.info("[lifespan] boot-time ingest inserted %d news items", count)
    except Exception:
        logger.exception("[lifespan] boot-time ingest failed")


def _format_validation_message(errors: list[Any]) -> str:
    parts: list[str] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        loc = err.get("loc") or ()
        loc_s = " → ".join(str(x) for x in loc if str(x) not in ("body", "query", "path"))
        msg = err.get("msg", "")
        if loc_s:
            parts.append(f"{loc_s}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts) if parts else "Invalid request"


app = FastAPI(
    title="Artie: Agentic Remote Trader that's Intelligently Engineered.",
    version="0.1.0",
    description="Artie is a remote trader that assists you in manageing your investment from voice commands on your smartphone!",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """422 bodies: keep Pydantic ``detail`` for tools, add plain ``message`` for UIs."""
    body = exc.errors()
    return JSONResponse(
        status_code=422,
        content={
            "detail": body,
            "message": _format_validation_message(body),
        },
    )

# Comma-separated list, e.g. CORS_ORIGINS=https://app.example.com,http://localhost:5173
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "").split(",")
    if o.strip()
]
_cors_cred = os.getenv("CORS_ALLOW_CREDENTIALS", "").lower() in ("1", "true", "yes")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_cred and bool(_cors_origins),
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Requested-With",
        "X-User-Id",
        "apikey",
        "x-client-info",
        "Prefer",
    ],
)

# Mount routers under both "/api/v1" (canonical) and "" (bare) so the existing
# nginx config that strips "/api/v1" still works without code changes.
for _prefix in ("/api/v1", ""):
    app.include_router(plaid_router, prefix=_prefix)
    app.include_router(exchanges_router, prefix=_prefix)
    app.include_router(snaptrade_router, prefix=_prefix)
    app.include_router(news_router, prefix=_prefix)
    app.include_router(conversations_router, prefix=_prefix)
    app.include_router(skills_router, prefix=_prefix)
    app.include_router(profile_router, prefix=_prefix)
    app.include_router(recommendations_router, prefix=_prefix)

# Initialize Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# The new TinyFish client for the 2026 hackathon
tf_client = TinyFish(api_key=os.getenv("TINYFISH_API_KEY"))


@app.get("/")
def health_check():
    return {"status": "Artie is online", "version": "0.1.0"}


_PLAID_CALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Artie</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
  <p style="font-family:system-ui;padding:2rem;">Returning you to Artie\u2026</p>
  <script>
    var qs = window.location.search || '';
    window.location.replace('artie://plaid-callback' + qs);
  </script>
  <noscript>
    <p>Please <a id="fallback" href="artie://plaid-callback">tap here</a> to return to Artie.</p>
    <script>
      document.getElementById('fallback').href =
        'artie://plaid-callback' + (window.location.search || '');
    </script>
  </noscript>
</body>
</html>
"""


@app.get("/plaid-callback", include_in_schema=False)
async def plaid_link_https_callback(request: Request) -> HTMLResponse:
    """Plaid redirects the in-app browser here after Link (HTTPS ``PLAID_REDIRECT_URI``).

    The page immediately redirects the in-app browser to the ``artie://`` deep-link
    scheme, preserving Plaid's query string (``public_token``, ``oauth_state_id``, etc.)
    so the Expo app can read them and call ``/api/v1/plaid/exchange``.
    """
    q = request.query_params
    logger.info(
        "[plaid] GET /plaid-callback query_keys=%s has_public_token=%s has_error=%s",
        sorted(q.keys()),
        "public_token" in q,
        "error" in q,
    )
    return HTMLResponse(
        content=_PLAID_CALLBACK_HTML,
        status_code=200,
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


# Legacy ``POST /process-voice`` was removed in Phase 3. Voice uploads now go
# through ``POST /conversations/{id}/messages/voice``, which persists the
# transcript, runs the orchestrator, and returns a durable assistant message.
