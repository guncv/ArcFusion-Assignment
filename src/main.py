from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from src.config import config
from src.utils.except_handler import validation_exception_handler, response_validation_exception_handler
from src.utils.exception import ArcFusionException
from src.api.routes import api_router_v1
from src.infras.ingestion import ingestion_manager
from src.infras.database import postgres_database

api_config = config.get("api", {})

app = FastAPI(
    title=api_config.get("title", "ArcFusion API"),
    version=api_config.get("version", "1.0.0"),
    docs_url=api_config.get("prefix", "/api/v1") + '/docs',
    openapi_url=api_config.get("prefix", "/api/v1") + '/openapi.json'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router_v1, prefix=api_config.get("prefix", "/api/v1"))

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ResponseValidationError, response_validation_exception_handler)
app.add_exception_handler(ArcFusionException, lambda request, exc: exc.convert_to_JSONResponse())

@app.on_event("startup")
async def startup_event():
    """Initialize database and run auto-ingestion on startup."""
    # Initialize database connection
    postgres_database.initialize()

    # Create tables
    try:
        await postgres_database.create_tables()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")

    # Run auto-ingestion if enabled
    if ingestion_manager.enabled and ingestion_manager.on_startup:
        ingestion_result = await ingestion_manager.ingest_documents()

        if ingestion_result:
            if ingestion_result.get("status") == "success":
                files_processed = ingestion_result.get("files_processed", 0)
                print(f"✅ Auto-ingestion completed: {files_processed} files processed")
            else:
                error = ingestion_result.get("error", "Unknown error")
                print(f"❌ Auto-ingestion failed: {error}")
        else:
            print("⚠️  Auto-ingestion returned no result")
    else:
        print("ℹ️  Auto-ingestion disabled or not configured")