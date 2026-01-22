import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
try:
    from config import config
except ImportError:
    # Fallback for when running scripts or different paths
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PgDBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PgDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.database_url = config.DATABASE_URL
        self.min_conn = config.POSTGRES_MIN_POOL_SIZE
        self.max_conn = config.POSTGRES_MAX_POOL_SIZE
        self.pool = None
        # Pool initialization is now explicit via connect()
        self._initialized = True

    def connect(self):
        """Initialize the connection pool"""
        if self.pool is None:
            try:
                self.pool = psycopg2.pool.ThreadedConnectionPool(
                    self.min_conn,
                    self.max_conn,
                    dsn=self.database_url
                )
                logger.info(f"PostgreSQL connection pool created successfully for {self.database_url}")
            except (Exception, psycopg2.DatabaseError) as error:
                logger.error(f"Error while connecting to PostgreSQL: {error}")
                # Re-raise to let the caller handle startup failure
                raise

    def disconnect(self):
        """Close all connections in the pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")
            self.pool = None

    @contextmanager
    def get_connection(self):
        """
        Context manager to get a connection from the pool.
        Automatically returns the connection to the pool when done.
        """
        if not self.pool:
            logger.warning("Connection pool not initialized, attempting to connect...")
            self.connect()

        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as error:
            logger.error(f"Error getting connection: {error}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def execute_query(self, query, params=None, fetch=False):
        """
        Execute a query using a connection from the pool.
        
        Args:
            query (str): SQL query to execute
            params (tuple, optional): Parameters for the query
            fetch (bool): Whether to fetch results (default: False)
            
        Returns:
            list: List of results if fetch is True, else None
        """
        result = None
        with self.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    if fetch:
                        result = cursor.fetchall()
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Query execution failed: {e}")
                raise e
        return result

# Create a global instance
pg_manager = PgDBManager()
