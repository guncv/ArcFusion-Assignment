from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from core.config.config import nested_config as config
from infrastructure.database.models import Base
import logging

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.async_session_factory = None
        self._initialize()

    def _initialize(self):
        try:
            db_config = config["database"]

            user = db_config["user"]
            password = db_config["password"]
            host = db_config["host"]
            port = db_config["port"]
            database = db_config["database"]

            database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

            self.engine = create_async_engine(
                database_url,
            )

            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info(f"Database connection initialized: {host}:{port}/{database}")

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    async def create_tables(self):
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    async def get_session(self) -> AsyncSession:
        async with self.async_session_factory() as session:
            yield session

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

db_connection = DatabaseConnection()