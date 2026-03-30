#!/usr/bin/env python3
"""Настройка окружения Google Colab.

Загружает секреты, настраивает DVC и MLflow,
скачивает данные из Yandex Cloud.

Использование (в ноутбуке Colab):
    !python scripts/colab_setup.py
    !python scripts/colab_setup.py --no-pull
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def is_colab() -> bool:
    """Проверяем, запущены ли мы в Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_credentials_from_colab_secrets() -> dict:
    """Загрузить секреты из Colab Secrets.

    В Colab: Runtime → Manage secrets → добавить ключи.

    Ожидаемые секреты:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        BUCKET_NAME
        MLFLOW_TRACKING_URI (опционально)
    """
    secrets = {}

    try:
        from google.colab import userdata

        secret_keys = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "BUCKET_NAME",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT",
        ]

        for key in secret_keys:
            try:
                value = userdata.get(key)
                if value:
                    os.environ[key] = value
                    secrets[key] = value
                    # Маскируем значение в выводе
                    display_val = value[:4] + "***" if len(value) > 4 else "***"
                    print(f"  ✓ {key} = {display_val}")
            except Exception:
                print(f"  ✗ {key} — не найден в Colab secrets")

    except ImportError:
        print("  ⚠ Не Colab, пропускаем secrets")

    return secrets


def setup_credentials_from_env_file(env_path: str = ".env") -> dict:
    """Запасной вариант: загрузить секреты из .env файла."""
    secrets = {}
    path = Path(env_path)

    if not path.exists():
        print(f"  ⚠ {env_path} не найден")
        return secrets

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
                secrets[key.strip()] = value.strip()

    print(f"  ✓ Загружено {len(secrets)} переменных из {env_path}")
    return secrets


def setup_dvc() -> bool:
    """Настроить DVC remote для Yandex Cloud S3."""
    bucket = os.environ.get("BUCKET_NAME")
    if not bucket:
        print("  ✗ BUCKET_NAME не задан, DVC не настроен")
        return False

    try:
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True, text=True,
        )

        if "storage" not in result.stdout:
            subprocess.run([
                "dvc", "remote", "add", "-d", "storage",
                f"s3://{bucket}/dvc-storage",
            ], check=True, capture_output=True)

            subprocess.run([
                "dvc", "remote", "modify", "storage",
                "endpointurl", "https://storage.yandexcloud.net",
            ], check=True, capture_output=True)

            print(f"  ✓ DVC remote настроен: s3://{bucket}/dvc-storage")
        else:
            print("  ✓ DVC remote уже настроен")

        return True

    except FileNotFoundError:
        print("  ✗ DVC не установлен, устанавливаем...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "dvc[s3]"],
            check=True, capture_output=True,
        )
        return setup_dvc()

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Ошибка DVC: {e}")
        return False


def pull_data() -> bool:
    """Скачать данные из DVC remote."""
    try:
        print("  Скачиваем данные (может занять несколько минут)...")
        subprocess.run(["dvc", "pull"], check=True)
        print("  ✓ Данные скачаны")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Ошибка dvc pull: {e}")
        return False


def setup_mlflow() -> bool:
    """Настроить MLflow трекинг."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("  ⚠ MLFLOW_TRACKING_URI не задан, используем файловый логгер")
        return False

    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        experiment = os.environ.get(
            "MLFLOW_EXPERIMENT", "road-surface-classification"
        )
        mlflow.set_experiment(experiment)
        print(f"  ✓ MLflow: {uri}")
        print(f"  ✓ Эксперимент: {experiment}")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка MLflow: {e}")
        return False


def check_gpu() -> None:
    """Проверить доступность GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  ✓ GPU: {gpu_name} ({gpu_mem:.1f} ГБ)")
        else:
            print("  ⚠ GPU недоступна (обучение будет медленным)")
            print("    В Colab: Runtime → Change runtime type → T4 GPU")
    except ImportError:
        print("  ✗ PyTorch не установлен")


def verify_data() -> None:
    """Проверить что данные скачаны."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("  ✗ Папка data/ не найдена")
        return

    for subdir in ["raw", "processed"]:
        path = data_dir / subdir
        if path.exists():
            file_count = len([f for f in path.rglob("*") if f.is_file()])
            print(f"  ✓ data/{subdir}: {file_count} файлов")
        else:
            print(f"  ✗ data/{subdir} не найден (запустите dvc pull)")

    # Проверяем CSV
    for csv_name in ["train.csv", "val.csv", "test.csv"]:
        csv_path = data_dir / "processed" / csv_name
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"  ✓ {csv_name}: {len(df)} сэмплов")
        else:
            print(f"  ✗ {csv_name} не найден")


def main():
    parser = argparse.ArgumentParser(description="Настройка окружения Colab")
    parser.add_argument(
        "--no-pull", action="store_true",
        help="Не скачивать данные",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("НАСТРОЙКА ОКРУЖЕНИЯ")
    print("=" * 60)

    print("\n[1/5] Проверка GPU...")
    check_gpu()

    print("\n[2/5] Загрузка секретов...")
    if is_colab():
        secrets = setup_credentials_from_colab_secrets()
        if not secrets:
            print("  Пробуем .env файл...")
            setup_credentials_from_env_file()
    else:
        setup_credentials_from_env_file()

    print("\n[3/5] Настройка DVC...")
    dvc_ok = setup_dvc()

    if not args.no_pull and dvc_ok:
        print("\n[4/5] Скачивание данных...")
        pull_data()
    else:
        print("\n[4/5] Пропускаем скачивание данных")

    print("\n[5/5] Настройка MLflow...")
    setup_mlflow()

    print("\n" + "-" * 60)
    print("ПРОВЕРКА")
    print("-" * 60)
    verify_data()

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print("\nДля обучения выполните:")
    print("  !python scripts/train.py --config configs/audio/models/simple_cnn.yaml")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()