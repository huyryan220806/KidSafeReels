"""Run the local API and web with one command; Ctrl+C shuts both down."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"
WEB_URL = "http://127.0.0.1:8501"


def port_in_use(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def app_is_running(backend):
    """Recognize the local API, requested backend and responding Streamlit web."""
    # Local checks must not go through a proxy configured in the user's shell.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"{API_URL}/health", timeout=1) as response:
            health = json.load(response)
        with opener.open(f"{API_URL}/openapi.json", timeout=1) as response:
            spec = json.load(response)
        with opener.open(f"{WEB_URL}/_stcore/health", timeout=1) as response:
            web_health = response.read().strip()
        return (
            health.get("status") == "ok"
            and health.get("backend") == backend
            and spec.get("info", {}).get("title") == "KidSafeReels"
            and "post" in spec.get("paths", {}).get("/v1/analyze", {})
            and web_health == b"ok"
        )
    except (OSError, URLError, ValueError, AttributeError, TypeError):
        return False


def open_web():
    try:
        if webbrowser.open(WEB_URL):
            return
    except (OSError, webbrowser.Error):
        pass
    print(f"Hãy mở trình duyệt tại {WEB_URL}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-browser", action="store_true", help="Mở web khi app sẵn sàng.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    backend = os.environ.get("KIDSAFE_BACKEND", "mock")
    occupied = [port for port in (8000, 8501) if port_in_use(port)]
    if occupied:
        if app_is_running(backend):
            print(f"KidSafeReels ({backend}) đang chạy. Mở lại web: {WEB_URL}", flush=True)
            if args.open_browser:
                open_web()
            return 0
        print(
            f"Cổng {', '.join(map(str, occupied))} đang được sử dụng, nhưng chưa xác nhận được "
            f"đủ API/web KidSafeReels với backend {backend}.\n"
            "Nếu app vừa khởi động, chờ vài giây rồi thử lại. Nếu đang chạy phiên khác, "
            "nhấn Ctrl+C ở terminal phiên đó rồi mở lại. Không tự dừng chương trình đang chiếm cổng.",
            file=sys.stderr,
        )
        return 1
    environment = os.environ.copy()
    environment["KIDSAFE_API_URL"] = API_URL
    environment["PYTHONIOENCODING"] = "utf-8"
    web = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "apps/web/app.py",
         "--server.address=127.0.0.1", "--server.port=8501", "--server.headless=true"],
        cwd=ROOT, env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    server = uvicorn.Server(uvicorn.Config("apps.api.main:app", host="127.0.0.1", port=8000, limit_concurrency=16))
    stopping = threading.Event()

    def watch_web():
        browser_pending = args.open_browser
        browser_deadline = time.monotonic() + 30
        while not stopping.wait(1):
            if web.poll() is not None:
                print("Web đã dừng. Đang đóng API.", flush=True)
                server.should_exit = True
                return
            if browser_pending:
                if server.started and app_is_running(backend):
                    open_web()
                    browser_pending = False
                elif time.monotonic() >= browser_deadline:
                    print(f"Nếu trình duyệt chưa mở, truy cập {WEB_URL}", flush=True)
                    browser_pending = False

    watcher = threading.Thread(target=watch_web, daemon=True)
    watcher.start()
    print("Web: http://127.0.0.1:8501 | API: http://127.0.0.1:8000/docs\nNhấn Ctrl+C để dừng cả hai dịch vụ.", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        if web.poll() is None:
            web.terminate()
            try:
                web.wait(timeout=5)
            except subprocess.TimeoutExpired:
                web.kill()
                web.wait()
        watcher.join(timeout=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
