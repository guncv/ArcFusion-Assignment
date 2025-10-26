from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config import config
from src.models import ChatMessage, EvaluationMetrics
from src.infras.log import logger

class PostgresDatabase:

    def __init__(self):
        self.engine = None
        self.async_session_factory = None

    def initialize(self):
        if self.engine is not None:
            return

        try:
            db_config = config["database"]
            user = db_config["user"]
            password = db_config["password"]
            host = db_config["host"]
            port = db_config["port"]
            database = db_config["database"]

            database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

            self.engine = create_async_engine(database_url)
            self.async_session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info(f"✅ Database engine initialized for {database}@{host}:{port}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise

    async def create_tables(self):
        if not self.engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(ChatMessage.metadata.create_all)
                await conn.run_sync(EvaluationMetrics.metadata.create_all)

            logger.info("✅ Database tables created successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise

    async def get_session(self):
        if not self.async_session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self.async_session_factory() as session:
            yield session

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            logger.info("🔒 Database connection closed.")
            self.engine = None
            self.async_session_factory = None

postgres_database = PostgresDatabase()
