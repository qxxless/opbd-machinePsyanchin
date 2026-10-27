"""Launch the Movie Recommendation API and Streamlit app for a demo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
APP_URL = "http://127.0.0.1:8501"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def request_status(url: str, timeout: float = 2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, ""


def wait_for_api(process: subprocess.Popen | None, timeout: int = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = request_status(f"{API_URL}/health")
        if status == 200:
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"API завершился с кодом {process.returncode}.")
        time.sleep(1)
    raise TimeoutError("API не ответил за 60 секунд. Проверьте MySQL и логи запуска.")


def wait_for_app(process: subprocess.Popen | None, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = request_status(APP_URL)
        if status == 200:
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Streamlit завершился с кодом {process.returncode}.")
        time.sleep(1)
    raise TimeoutError("Streamlit не ответил за 60 секунд.")


def port_is_in_use(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Запустить API и приложение рекомендаций.")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер автоматически.")
    args = parser.parse_args()

    if not (ROOT / ".env").exists():
        raise SystemExit("Не найден .env. Сначала создайте его из .env.example и укажите подключение к MySQL.")
    if not (ROOT / "models" / "recommender_bundle.joblib").exists():
        raise SystemExit("Не найдена сохранённая модель models/recommender_bundle.joblib.")
    if not VENV_PYTHON.exists():
        raise SystemExit("Не найдено окружение .venv. Установите зависимости по инструкции в README.md.")

    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

    api_process = None
    app_process = None
    try:
        api_status, _ = request_status(f"{API_URL}/health")
        if api_status != 200:
            if port_is_in_use(8000):
                raise RuntimeError("Порт 8000 занят другим процессом. Закройте его или остановите старый API.")
            api_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=ROOT,
            )
        health = wait_for_api(api_process)
        if not health.get("database_connected"):
            raise RuntimeError("API запустился, но MySQL недоступен. Запустите MySQL Server и проверьте .env.")

        app_status, _ = request_status(APP_URL)
        if app_status != 200:
            if port_is_in_use(8501):
                raise RuntimeError("Порт 8501 занят другим процессом. Закройте старый Streamlit.")
            app_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                 "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true"],
                cwd=ROOT,
            )
        wait_for_app(app_process)

        print("API и MySQL готовы:", health)
        print("Приложение:", APP_URL)
        print("Документация API:", f"{API_URL}/docs")
        print("Нажмите Ctrl+C, чтобы остановить процессы, запущенные этим скриптом.")
        if not args.no_browser:
            webbrowser.open(APP_URL)

        while True:
            for label, process in (("API", api_process), ("Streamlit", app_process)):
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"{label} завершился с кодом {process.returncode}.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nОстанавливаю запущенные сервисы...")
        return 0
    finally:
        stop_process(app_process)
        stop_process(api_process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        raise SystemExit(1)
