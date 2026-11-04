"""The HCM database URL is remembered in <projects_root>/settings.json (owner-only), so later starts
need no export. python -m pytest tests/test_hcm_settings.py"""
import json
import os
import stat
import tempfile
from pathlib import Path

from laras_labeler import config, hcm_roi


def test_env_url_is_remembered_then_used_without_env(monkeypatch):
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "")
    root = Path(tempfile.mkdtemp())
    monkeypatch.setenv("HCM_DATABASE_URL", "postgresql://u:p@h:5432/db")
    assert hcm_roi.use_database_url(root) == "env" and hcm_roi.DATABASE_URL == "postgresql://u:p@h:5432/db"
    p = config.settings_path(root)
    assert json.loads(p.read_text())[hcm_roi.SETTINGS_KEY] == "postgresql://u:p@h:5432/db"
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o600          # it holds a password: owner only
    monkeypatch.delenv("HCM_DATABASE_URL")
    hcm_roi.DATABASE_URL = ""
    assert hcm_roi.use_database_url(root) == "saved" and hcm_roi.DATABASE_URL == "postgresql://u:p@h:5432/db"
    monkeypatch.setenv("HCM_DATABASE_URL", "postgresql://u:p@other:5432/db")   # a new export replaces it
    hcm_roi.use_database_url(root)
    assert json.loads(p.read_text())[hcm_roi.SETTINGS_KEY].endswith("@other:5432/db")


def test_other_settings_survive_and_none_means_none(monkeypatch):
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "")
    root = Path(tempfile.mkdtemp())
    config.save_app_settings(root, {"hidra_home": "/x/HiDRA"})
    monkeypatch.delenv("HCM_DATABASE_URL", raising=False)
    assert hcm_roi.use_database_url(root) == "" and hcm_roi.DATABASE_URL == ""
    monkeypatch.setenv("HCM_DATABASE_URL", "postgresql://u:p@h/db")
    hcm_roi.use_database_url(root)
    assert config.load_app_settings(root)["hidra_home"] == "/x/HiDRA"


def test_the_url_never_reaches_the_browser(monkeypatch):
    monkeypatch.setattr(hcm_roi, "DATABASE_URL", "")
    from fastapi.testclient import TestClient
    from laras_labeler.app import create_app
    from laras_labeler.config import Settings
    from laras_labeler.project import ProjectStore
    root = Path(tempfile.mkdtemp())
    monkeypatch.setenv("HCM_DATABASE_URL", "postgresql://u:SECRETPW@h/db")
    c = TestClient(create_app(Settings(projects_root=root), ProjectStore(root)))
    pid = c.post("/api/projects", json={"name": "t"}).json()["pid"]
    for path in ("/api/hidra/status", f"/api/projects/{pid}", "/api/projects"):
        r = c.get(path)
        assert "SECRETPW" not in r.text, path
