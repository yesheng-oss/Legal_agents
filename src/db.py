from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from settings import get_settings


def create_session_factory(database_url=None):
    url = database_url or get_settings().database_url
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    elif url.startswith("postgresql"):
        connect_args = {"connect_timeout": 3}
    else:
        connect_args = {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def is_database_available(database_url=None, timeout=1):
    url = database_url or get_settings().database_url
    if url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    elif url.startswith("postgresql"):
        connect_args = {"connect_timeout": timeout}
    else:
        connect_args = {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    try:
        with engine.connect() as connection:
            connection.execute(text("select 1"))
        return True
    except Exception:
        return False
    finally:
        engine.dispose()


def init_db(metadata, session_factory):
    metadata.create_all(session_factory.kw["bind"])


@contextmanager
def session_scope(session_factory):
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
