"""
FactoryGuard AI — API Demo Script
Member 3 (Nikhil) | demo/demo_requests.py

Sends 3 test requests to the /predict endpoint:
  Scenario 1 — Healthy machine        → low failure probability
  Scenario 2 — Machine at 50% life    → medium failure probability
  Scenario 3 — Machine near failure   → high failure probability (> 0.8)

Usage:
    Make sure the Flask API is running first:
        python api/app.py

    Then run this script:
        python demo/demo_requests.py
"""

import requests
import json

API_URL = "http://localhost:5000/predict"

# ──────────────────────────────────────────────
# Helper — build a sensor reading dict
# The API expects the same feature columns that
# src/feature_engineering.py produces.
# Here we send the 21 raw sensor values; the API
# internally applies rolling/lag features via the
# saved preprocessor before predicting.
# ──────────────────────────────────────────────

def make_sensor_reading(health_level: str) -> dict:
    """
    Returns a dict of 21 sensor values mimicking a
    NASA CMAPSS turbofan engine at different health levels.

    health_level: 'healthy' | 'mid_life' | 'near_failure'
    """

    # Sensor baseline values for a healthy engine (cycle ~10)
    # Values derived from typical FD001 dataset means.
    healthy = {
        "s1":  518.67, "s2":  641.82, "s3": 1589.70, "s4": 1400.60,
        "s5":   14.62, "s6":   21.61, "s7":  554.36,  "s8": 2388.06,
        "s9":  9046.19,"s10":  1.30,  "s11":  47.47,  "s12":  521.66,
        "s13": 2388.02,"s14": 8138.62,"s15":   8.42,  "s16":   0.03,
        "s17":  392.00,"s18": 2388.00,"s19":  100.00, "s20":   39.06,
        "s21":   23.42
    }

    # Mid-life engine: sensors drift slightly (cycle ~100)
    mid_life = {
        "s1":  518.67, "s2":  642.58, "s3": 1582.15, "s4": 1392.30,
        "s5":   14.62, "s6":   21.61, "s7":  551.20,  "s8": 2388.06,
        "s9":  9044.10,"s10":  1.30,  "s11":  47.91,  "s12":  518.40,
        "s13": 2387.72,"s14": 8125.55,"s15":   8.50,  "s16":   0.03,
        "s17":  391.00,"s18": 2388.00,"s19":  100.00, "s20":   38.88,
        "s21":   23.15
    }

    # Near-failure engine: sensors degrade significantly (cycle ~190+)
    near_failure = {
        "s1":  518.67, "s2":  644.90, "s3": 1568.40, "s4": 1375.80,
        "s5":   14.62, "s6":   21.61, "s7":  545.11,  "s8": 2388.06,
        "s9":  9032.00,"s10":  1.30,  "s11":  48.92,  "s12":  511.00,
        "s13": 2387.10,"s14": 8098.70,"s15":   8.72,  "s16":   0.03,
        "s17":  388.00,"s18": 2388.00,"s19":  100.00, "s20":   38.20,
        "s21":   22.60
    }

    mapping = {
        "healthy":      healthy,
        "mid_life":     mid_life,
        "near_failure": near_failure,
    }
    return mapping[health_level]


def send_request(scenario_name: str, sensor_data: dict) -> None:
    """Send one POST request and print the result."""
    print(f"\n{'='*55}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*55}")

    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(sensor_data),
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        prob = result.get("failure_probability", "N/A")
        label = result.get("prediction_label", "N/A")

        print(f"  failure_probability : {prob}")
        print(f"  prediction_label    : {label}")
        print(f"  Full response       : {result}")

        # Sanity check
        if isinstance(prob, float):
            if scenario_name == "Scenario 1 — Healthy Machine":
                assert prob < 0.3, f"Expected low prob, got {prob:.3f}"
                print("  ✓ PASS — probability is low as expected")
            elif scenario_name == "Scenario 2 — Machine at 50% Life":
                assert 0.3 <= prob <= 0.75, f"Expected medium prob, got {prob:.3f}"
                print("  ✓ PASS — probability is medium as expected")
            elif scenario_name == "Scenario 3 — Machine Near Failure":
                assert prob > 0.75, f"Expected high prob (>0.75), got {prob:.3f}"
                print("  ✓ PASS — probability is high as expected")

    except requests.exceptions.ConnectionError:
        print("  ✗ ERROR — Could not connect. Is 'python api/app.py' running?")
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP ERROR — {e}")
    except AssertionError as e:
        print(f"  ✗ UNEXPECTED RESULT — {e}")
    except Exception as e:
        print(f"  ✗ UNEXPECTED ERROR — {e}")


# ──────────────────────────────────────────────
# Main — run all 3 scenarios
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\nFactoryGuard AI — API Demo")
    print("Testing 3 scenarios against:", API_URL)

    # Scenario 1: Healthy machine — expect LOW failure probability
    send_request(
        "Scenario 1 — Healthy Machine",
        make_sensor_reading("healthy")
    )

    # Scenario 2: Mid-life machine — expect MEDIUM failure probability
    send_request(
        "Scenario 2 — Machine at 50% Life",
        make_sensor_reading("mid_life")
    )

    # Scenario 3: Near-failure machine — expect HIGH failure probability > 0.8
    send_request(
        "Scenario 3 — Machine Near Failure",
        make_sensor_reading("near_failure")
    )

    print(f"\n{'='*55}")
    print("  Demo complete.")
    print(f"{'='*55}\n")
