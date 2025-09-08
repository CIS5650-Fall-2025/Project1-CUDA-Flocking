import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent.absolute()
BUILD_DIR = ROOT_DIR / "build"
BOIDS_EXE = BUILD_DIR / "bin" / "Release" / "cis5650_boids.exe"
MEASUREMENTS_JSON = Path(__file__).parent.absolute() / "measurements.json"


def measure_fps_vs_num_boids() -> None:
    measurement_data: list[dict[str, Any]] = []
    if MEASUREMENTS_JSON.exists():
        assert MEASUREMENTS_JSON.is_file()
        with MEASUREMENTS_JSON.open(encoding="utf-8") as file:
            measurement_data = json.load(file)
    else:
        measurement_data = []

    def find_measurement(config: dict[str, Any]) -> dict[str, Any] | None:
        for measurement in measurement_data:
            if all(
                key in measurement and measurement[key] == value
                for key, value in config.items()
            ):
                return measurement
        return None

    nums_boids = [5000 * (2**i) for i in range(13)]
    for name, base_config in [
        ("naive", {"UNIFORM_GRID": 0, "COHERENT_GRID": 0}),
        ("scattered uniform grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 0}),
        ("coherent uniform grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 1}),
    ]:
        print(f"Measuring FPS for method: {name}")
        for visualize in [0, 1]:
            for num_boids in nums_boids:
                config = {
                    "N_FOR_VIS": num_boids,
                    "VISUALIZE": visualize,
                    **base_config,
                    "CUDA_BLOCK_SIZE": 128,
                }
                measurement = find_measurement(config)
                if measurement is None:
                    print(f"Measuring FPS for config: {config}")
                    start_time = time.time()
                    fps = build_and_measure_fps(**config)
                    duration = time.time() - start_time
                    measurement = {**config, "duration": duration, "fps": fps}
                    measurement_data.append(measurement)
                    with MEASUREMENTS_JSON.open("w", encoding="utf-8") as file:
                        json.dump(measurement_data, file, indent=4)
                else:
                    print(f"Found existing measurement for config: {config}")
                print(f"  Duration: {measurement['duration']}")
                print(f"  FPS: {measurement['fps']}")
                if measurement["duration"] > 100.0:
                    print("Skipping further measurements due to long duration")
                    break
                if measurement["fps"] < 1.0:
                    print("Skipping further measurements due to low FPS")
                    break


def measure_fps_vs_block_size() -> None:
    measurement_data: list[dict[str, Any]] = []
    if MEASUREMENTS_JSON.exists():
        assert MEASUREMENTS_JSON.is_file()
        with MEASUREMENTS_JSON.open(encoding="utf-8") as file:
            measurement_data = json.load(file)
    else:
        measurement_data = []

    def find_measurement(config: dict[str, Any]) -> dict[str, Any] | None:
        for measurement in measurement_data:
            if all(
                key in measurement and measurement[key] == value
                for key, value in config.items()
            ):
                return measurement
        return None

    for name, base_config in [
        ("naive", {"UNIFORM_GRID": 0, "COHERENT_GRID": 0}),
        ("scattered uniform grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 0}),
        ("coherent uniform grid", {"UNIFORM_GRID": 1, "COHERENT_GRID": 1}),
    ]:
        print(f"Measuring FPS for method: {name}")
        for block_size in [8 * (2**i) for i in range(8)]:
            config = {
                "N_FOR_VIS": 320000,
                "VISUALIZE": 0,
                **base_config,
                "CUDA_BLOCK_SIZE": block_size,
            }
            measurement = find_measurement(config)
            if measurement is None:
                print(f"Measuring FPS for config: {config}")
                start_time = time.time()
                fps = build_and_measure_fps(**config)
                duration = time.time() - start_time
                measurement = {**config, "duration": duration, "fps": fps}
                measurement_data.append(measurement)
                with MEASUREMENTS_JSON.open("w", encoding="utf-8") as file:
                    json.dump(measurement_data, file, indent=4)
            else:
                print(f"Found existing measurement for config: {config}")
            print(f"  Duration: {measurement['duration']}")
            print(f"  FPS: {measurement['fps']}")


def build_and_measure_fps(**kwargs: str) -> float:
    build(**kwargs)
    return measure_fps()


def build(**kwargs: str) -> None:
    if BUILD_DIR.exists():
        assert BUILD_DIR.is_dir()
        shutil.rmtree(BUILD_DIR)

    subprocess.run(
        args=[
            "cmake",
            "-G",
            "Visual Studio 17 2022",
            "-S",
            str(ROOT_DIR),
            "-B",
            str(BUILD_DIR),
            *(
                f"-D{key}={value}"
                for key, value in {
                    **kwargs,
                    "FPS_MEASURE": 1,
                    "FPS_MEASURE_START": 2,
                    "FPS_MEASURE_DURATION": 20,
                }.items()
            ),
        ],
        check=True,
    )
    assert BUILD_DIR.is_dir()

    subprocess.run(
        [
            "cmake",
            "--build",
            str(BUILD_DIR),
            "--target",
            "cis5650_boids",
            "--config",
            "Release",
            "--parallel",
        ],
        check=True,
    )
    assert BOIDS_EXE.is_file()


def measure_fps() -> float:
    assert BOIDS_EXE.is_file()
    result = subprocess.run(
        [str(BOIDS_EXE)], cwd=BUILD_DIR, capture_output=True, check=True, text=True
    )
    for line in result.stdout.splitlines():
        re_match = re.match(r"^FPS: (.+)$", line)
        if re_match is not None:
            return float(re_match[1])
    raise RuntimeError("FPS not found in output")


if __name__ == "__main__":
    measure_fps_vs_num_boids()
    measure_fps_vs_block_size()
