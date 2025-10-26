from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from core.config.config import api_config
from core.utils.except_handler import validation_exception_handler, response_validation_exception_handler
from core.utils.exception import ArcFusionException
from app.router import api_router_v1
from infrastructure.rag.auto_ingestion import ingestion_manager
from infrastructure.database.connection import db_connection

app = FastAPI(
    title=api_config.get("API_TITLE", "ArcFusion API"),
    version=api_config.get("API_VERSION", "1.0.0"),
    docs_url=api_config.get("API_PREFIX", "/api/v1")+'/docs',
    openapi_url=api_config.get("API_PREFIX", "/api/v1")+'/openapi.json'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router_v1, prefix=api_config.get("API_PREFIX", "/api/v1"))

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ResponseValidationError, response_validation_exception_handler)
app.add_exception_handler(ArcFusionException, lambda request, exc: exc.convert_to_JSONResponse())

@app.on_event("startup")
async def startup_event():
    try:
        await db_connection.create_tables()
    except Exception as e:
        print(f"Failed to create database tables: {e}")
    
    ingestion_result = await ingestion_manager.ingest_documents()

    if ingestion_result:
        if ingestion_result.get("status") == "success":
            print("Auto-ingestion completed successfully!")
        else:
            print("Auto-ingestion failed - check logs for details")
    else:
        print("Auto-ingestion skipped (not configured or no documents found)")