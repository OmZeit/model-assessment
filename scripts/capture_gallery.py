"""
Automated screenshot and demo GIF capture for AssayReady 2.0.
Uses headless Chrome via Selenium and stitches frames using Pillow.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8050"


def setup_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1600,1000")
    options.add_argument("--hide-scrollbars")
    return webdriver.Chrome(options=options)


def main():
    print(f"Connecting to {BASE_URL}...")
    driver = setup_driver()
    driver.set_page_load_timeout(30)

    frames = []

    def record_frame(name: str | None = None, pause: float = 1.0):
        time.sleep(pause)
        temp_path = OUTPUT_DIR / "_temp_frame.png"
        driver.save_screenshot(str(temp_path))
        img = Image.open(temp_path).convert("RGB")
        gif_img = img.resize((1120, 700), Image.Resampling.LANCZOS)
        frames.append(gif_img)
        if name:
            img.save(OUTPUT_DIR / f"{name}.png", optimize=True)
            print(f"  -> Saved {name}.png ({img.size[0]}x{img.size[1]})")

    try:
        # 1. Analyze initial screen
        print("1. Capturing Initial Analyze Canvas...")
        driver.get(f"{BASE_URL}/analyze")
        record_frame("console_overview", pause=3.0)

        # 2. Load sample data
        print("2. Loading sample data...")
        try:
            sample_btn = driver.find_element(By.ID, "load-prediction-example-button")
            sample_btn.click()
            record_frame(pause=2.5)
        except Exception as e:
            print(f"Error clicking sample button: {e}")

        # 3. Confirm mappings and run
        print("3. Confirming mappings and running analysis...")
        try:
            confirm_btn = driver.find_element(By.ID, "confirm-mappings-button")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", confirm_btn)
            time.sleep(1)
            confirm_btn.click()
            record_frame(pause=2.0)

            run_btn = driver.find_element(By.ID, "run-analysis-button")
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", run_btn)
            time.sleep(1)
            run_btn.click()
            time.sleep(4)
            driver.execute_script("window.scrollTo(0, 380);")
            record_frame("analysis_evidence_package", pause=2.0)
        except Exception as e:
            print(f"Error running analysis: {e}")

        # 4. Candidate explorer
        print("4. Capturing Candidate Explorer...")
        driver.get(f"{BASE_URL}/candidates?run_id=beaa9e0b7ebf42e392d377829190e95d")
        record_frame("candidate_explorer", pause=3.0)

        # 5. Design Sandbox
        print("5. Capturing Design Sandbox...")
        driver.get(f"{BASE_URL}/simulations")
        record_frame("design_sandbox", pause=3.0)

        # 6. Generate draft designs
        print("6. Generating draft designs in Sandbox...")
        try:
            gen_btn = driver.find_element(By.ID, "run-simulation-button")
            gen_btn.click()
            time.sleep(4)
            driver.execute_script("window.scrollTo(0, 480);")
            record_frame("design_sandbox_generated", pause=2.0)
        except Exception as e:
            print(f"Error in simulation generation: {e}")

        # 7. Benchmark Library
        print("7. Capturing Benchmark Library...")
        driver.get(f"{BASE_URL}/benchmarks")
        record_frame("benchmarks_library", pause=3.0)

        # 8. Evidence Reports
        print("8. Capturing Evidence Reports...")
        driver.get(f"{BASE_URL}/reports?run_id=beaa9e0b7ebf42e392d377829190e95d")
        record_frame("evidence_report", pause=3.0)

        # Clean up temp file
        temp_frame = OUTPUT_DIR / "_temp_frame.png"
        if temp_frame.is_file():
            temp_frame.unlink()

        # Build animated demo GIF
        if frames:
            gif_path = OUTPUT_DIR / "assayready_demo.gif"
            print(f"Creating animated GIF with {len(frames)} frames at {gif_path}...")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=2200,
                loop=0,
                optimize=True,
            )
            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"  -> Generated {gif_path.name} ({gif_size_mb:.2f} MB)")

    finally:
        driver.quit()
        print("Done capturing gallery!")


if __name__ == "__main__":
    main()
