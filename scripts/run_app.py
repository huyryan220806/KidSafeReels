"""Run the local API and web with one command; Ctrl+C shuts both down."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def check_port(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Cổng {port} đang được sử dụng. Dừng phiên app cũ trước khi chạy lại.") from exc


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    for port in (8000, 8501):
        check_port(port)
    environment = os.environ.copy()
    environment["KIDSAFE_API_URL"] = "http://127.0.0.1:8000"
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
        while not stopping.wait(1):
            if web.poll() is not None:
                print("Web đã dừng. Đang đóng API.", flush=True)
                server.should_exit = True
                return

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


if __name__ == "__main__":
    main()
