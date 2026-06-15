from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
from sqlalchemy import create_engine, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import Base
from settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    return get_settings().database_url


def run_migrations_offline():
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connect_args = {}
    url = get_url()
    if url.startswith("postgresql"):
        connect_args = {"connect_timeout": 3}
    elif url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    connectable = create_engine(
        url,
        future=True,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
