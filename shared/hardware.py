"""Runtime hardware and environment detection helpers."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any, Dict


def detect_kaggle_environment() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "KAGGLE_KERNEL_RUN_TYPE",
            "KAGGLE_URL_BASE",
            "KAGGLE_PROJECT_DIR",
            "KAGGLE_CONTAINER_ID",
        )
    )


def detect_gpu(min_memory_gb: int = 6) -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {
        "available": False,
        "framework": None,
        "devices": [],
        "details": {},
        "enough": False,
        "min_memory_gb": min_memory_gb,
    }

    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            devices = []
            enough = False
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total_gb = round(props.total_memory / 1024**3, 2)
                devices.append(f"{props.name} ({total_gb} GB)")
                if total_gb >= min_memory_gb:
                    enough = True

            gpu_info["available"] = True
            gpu_info["framework"] = "torch"
            gpu_info["devices"] = devices
            gpu_info["details"] = {
                "count": count,
                "current_device": torch.cuda.current_device(),
                "total_memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2) for i in range(count)],
            }
            gpu_info["enough"] = enough
            return gpu_info
    except Exception:
        pass

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            devices = [gpu.name for gpu in gpus]
            gpu_info["available"] = True
            gpu_info["framework"] = "tensorflow"
            gpu_info["devices"] = devices
            gpu_info["details"] = {"count": len(gpus)}
            gpu_info["enough"] = True
            return gpu_info
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                devices = []
                enough = False
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",") if part.strip()]
                    if len(parts) >= 2:
                        name, memory = parts[0], parts[1]
                        memory_gb = round(float(memory) / 1024, 2)
                        devices.append(f"{name} ({memory_gb} GB)")
                        if memory_gb >= min_memory_gb:
                            enough = True
                if devices:
                    gpu_info["available"] = True
                    gpu_info["framework"] = "nvidia-smi"
                    gpu_info["devices"] = devices
                    gpu_info["details"] = {"count": len(devices)}
                    gpu_info["enough"] = enough
        except Exception:
            pass

    return gpu_info


def get_runtime_info() -> Dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "kaggle": detect_kaggle_environment(),
        "gpu": detect_gpu(),
    }
