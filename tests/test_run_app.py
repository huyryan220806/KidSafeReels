"""Launcher regressions: reopening a healthy app never starts duplicate services."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from scripts import run_app


@pytest.fixture
def local_services(monkeypatch):
    responses = {
        "/health": (200, json.dumps({"status": "ok", "backend": "model"})),
        "/openapi.json": (200, json.dumps({
            "info": {"title": "KidSafeReels"},
            "paths": {"/v1/analyze": {"post": {}}},
        })),
        "/_stcore/health": (200, "ok"),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = responses.get(self.path, (404, "missing"))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(run_app, "API_URL", url)
    monkeypatch.setattr(run_app, "WEB_URL", url)
    monkeypatch.setattr(run_app, "port_in_use", lambda port: True)
    monkeypatch.setenv("KIDSAFE_BACKEND", "model")
    monkeypatch.setattr(run_app.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate launch"))
    opened = []
    monkeypatch.setattr(run_app.webbrowser, "open", lambda url: opened.append(url) or True)
    yield responses, opened
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("open_browser", [False, True])
def test_reopen_healthy_app_without_spawning(local_services, open_browser, capsys):
    _, opened = local_services
    assert run_app.main(["--open-browser"] if open_browser else []) == 0
    assert opened == ([run_app.WEB_URL] if open_browser else [])
    assert "đang chạy" in capsys.readouterr().out


@pytest.mark.parametrize("path,response", [
    ("/health", (200, '{"status":"ok","backend":"mock"}')),
    ("/health", (200, '{"status":"error","backend":"model"}')),
    ("/health", (200, '[]')),
    ("/openapi.json", (200, '{"info":{"title":"Another app"}}')),
    ("/openapi.json", (200, '{"info":{"title":"KidSafeReels"},"paths":{}}')),
    ("/openapi.json", (200, '<html>not JSON</html>')),
    ("/_stcore/health", (503, 'unavailable')),
    ("/_stcore/health", (200, 'starting')),
])
def test_occupied_ports_do_not_reuse_wrong_or_unhealthy_service(local_services, path, response, capsys):
    responses, opened = local_services
    responses[path] = response
    assert run_app.main(["--open-browser"]) == 1
    assert opened == []
    assert "Cổng 8000, 8501" in capsys.readouterr().err


def test_browser_failure_keeps_existing_app_usable(local_services, monkeypatch, capsys):
    def fail_to_open(url):
        raise OSError("No default browser")

    monkeypatch.setattr(run_app.webbrowser, "open", fail_to_open)
    assert run_app.main(["--open-browser"]) == 0
    assert run_app.WEB_URL in capsys.readouterr().out


def test_fresh_launch_opens_after_readiness_and_cleans_up(monkeypatch):
    opened = threading.Event()
    launched = []
    health_checks = []

    class WebProcess:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            assert self.stopped
            return 0

    web = WebProcess()

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return web

    class Server:
        started = False

        def __init__(self, config):
            pass

        def run(self):
            self.started = True
            assert opened.wait(5), "Browser did not open when services became ready"

    def ready(backend):
        health_checks.append(backend)
        return len(health_checks) > 1

    monkeypatch.setenv("KIDSAFE_BACKEND", "model")
    monkeypatch.setattr(run_app, "port_in_use", lambda port: False)
    monkeypatch.setattr(run_app, "app_is_running", ready)
    monkeypatch.setattr(run_app, "open_web", opened.set)
    monkeypatch.setattr(run_app.subprocess, "Popen", launch)
    monkeypatch.setattr(run_app.uvicorn, "Server", Server)
    assert run_app.main(["--open-browser"]) == 0
    assert len(launched) == 1
    assert launched[0][1]["env"]["KIDSAFE_API_URL"] == run_app.API_URL
    assert health_checks == ["model", "model"]
    assert web.stopped
