import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import db


def test_postgres_session_factory_sets_short_connect_timeout(monkeypatch):
    captured = {}

    def fake_create_engine(url, future, connect_args):
        captured["url"] = url
        captured["future"] = future
        captured["connect_args"] = connect_args
        return object()

    def fake_sessionmaker(**kwargs):
        captured["sessionmaker_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    monkeypatch.setattr(db, "sessionmaker", fake_sessionmaker)

    db.create_session_factory("postgresql+psycopg://user:pass@localhost:5432/legal_agent")

    assert captured["connect_args"]["connect_timeout"] == 3
    assert captured["future"] is True
