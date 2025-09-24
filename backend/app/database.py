import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import psycopg2
from psycopg2.extensions import register_adapter, AsIs


load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("A variável de ambiente 'DATABASE_URL' não está definida.")


def adapt_list_for_pgvector(list_data):
    return AsIs(str(list_data).replace('[', '(').replace(']', ')'))

register_adapter(list, adapt_list_for_pgvector)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()