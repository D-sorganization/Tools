# Putting Launch Monitor — Accuracy Validation and Evidence

Document governing epic #5218, child issue #5221.

This document records the measurement methodology, calibration parameters,
reference speed and HLA sources, acceptance criteria, and validation evidence
for the camera-based putting launch monitor.

---

## 1. Acceptance Criteria

| Metric | Target Tolerance | Range |
| :--- | :--- | :--- |
| **Launch Speed** | Within **±3.0%** of reference | 1.0 – 4.0 mph (and extended to 8.0 mph) |
| **Horizontal Launch Angle (HLA)** | Within **±1.0°** of reference | -10.0° to +10.0° |
| **Putt Acceptance Rate** | > 95% on clean rolls | Clean unhindered rolls past 300 mm |
| **Negative Rejection** | 0 false putts | Hand placement, ball placement, rolling unarmed |

---

## 2. Reference Methodologies

### 2.1 Launch Speed Reference

Two independent physical methods establish ground-truth launch speed in the bay:

1. **Calibrated Gravity Ramp (Primary)**:
   - A rigid putting ramp released from fixed release heights $h$.
   - Theoretical release velocity:
     $$v = \sqrt{2 g h} - v_{\text{loss}}$$
   - **Ramp Calibration**: The ramp is calibrated once by timing the ball with an electronic optical gate or stopwatch over a taped 1.00 m distance on the level green surface immediately following ramp exit.
   - Three standard heights provide three repeatable release speeds:
     - Low: ~1.5 mph (~0.67 m/s)
     - Mid: ~2.5 mph (~1.12 m/s)
     - High: ~3.8 mph (~1.70 m/s)

2. **Roll-Out Distance on Known Stimp (Secondary)**:
   - Measures total roll-out distance $d$ on a level surface of measured Stimpmeter rating $S$ (feet).
   - Friction coefficient $\mu = \frac{g \cdot S_{\text{stimp\_ft}}}{d_{\text{stimp}}}$.
   - Initial velocity $v_0 = \sqrt{2 \mu g d}$.
   - Model implemented in `shared.python.swing_sim.putting.roll`.

### 2.2 Horizontal Launch Angle (HLA) Reference

1. **Alignment Guide Lines**:
   - High-contrast laser or fine chalk/tape lines marked on the hitting mat at known angles relative to the long axis ($y$-axis toward target):
     - Straight: $0.0^\circ$
     - Push: $+5.0^\circ, +10.0^\circ$ (player right)
     - Pull: $-5.0^\circ, -10.0^\circ$ (player left)
   - Ball is guided along the line using a grooved guide track.

---

## 3. Calibration and Bay Setup

The monitor relies on a ground-plane homography defined by four mat corners:

- **Corner Order** (player's perspective looking toward the target):
  1. Near-left
  2. Near-right
  3. Far-right
  4. Far-left
- **Mat Dimensions**:
  - Measured with precision tape across stance width and along target length in mm.
  - Example setup: $1220 \text{ mm} \times 1520 \text{ mm}$.
- **Camera Position**:
  - Overhead camera mounted ~2.4 m above the mat, tilted ~30° toward the player.
  - Native resolution: 1920x1200@60 fps; pipeline decode width: 960x600 px.

---

## 4. Running the Validation Harness

Use the `validate` CLI command to log putts and running statistics:

```bash
# Live camera with interactive prompt for operator reference values:
python -m putting_launch_monitor validate --camera "USB\VID_..." --out validation_bay.csv

# Replaying recorded video clips with fixed reference values:
python -m putting_launch_monitor validate --video clip.avi --ref-speed 3.50 --ref-hla 0.0 --out val.csv --non-interactive
```

The CSV output records:
`timestamp, speed_mph, hla_deg, points, r2, span_mm, ref_speed_mph, ref_hla_deg, speed_err_pct, hla_err_deg, accepted, reason, note`.

---

## 5. Validation Evidence and Results

### 5.1 Benchmark Accuracy Results (Lab Geometry)

| Test Case / Clip | Ref Speed (mph) | Meas Speed (mph) | Speed Error (%) | Ref HLA (deg) | Meas HLA (deg) | HLA Error (deg) | Fit $r^2$ | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Slow Straight** | 3.36 | 3.34 | -0.60% | 0.00° | -0.13° | -0.13° | 0.999 | **PASS** |
| **Medium Pull** | 4.47 | 4.44 | -0.67% | -2.50° | -2.60° | -0.10° | 0.999 | **PASS** |
| **Medium Push** | 5.14 | 5.15 | +0.19% | +2.00° | +2.13° | +0.13° | 0.999 | **PASS** |
| **Fast Straight** | 6.71 | 6.68 | -0.45% | 0.00° | +0.20° | +0.20° | 0.999 | **PASS** |
| **Fast Slight Pull** | 6.26 | 6.24 | -0.32% | -1.20° | -1.40° | -0.20° | 0.999 | **PASS** |

### 5.2 Summary Statistics

- **Mean Speed Error**: $-0.37\%$
- **Speed Mean Absolute Error (MAE)**: $0.45\%$ (well within $\pm 3.0\%$ gate)
- **Max Speed Error**: $0.67\%$
- **Mean HLA Error**: $-0.02^\circ$
- **HLA Mean Absolute Error (MAE)**: $0.15^\circ$ (well within $\pm 1.0^\circ$ gate)
- **Max HLA Error**: $0.20^\circ$

### 5.3 Negative Control Verification

| Negative Control Case | Action | Expected Outcome | Observed Result | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Hand Placing Ball** | Hand enters, places ball at rest, leaves | 0 accepted putts | 0 accepted putts (tracker arms at rest, no launch) | **PASS** |
| **Rolling Unarmed** | Ball rolled across mat without resting | 0 accepted putts | 0 accepted putts (tracker never arms, roll ignored) | **PASS** |

---

## 6. Rejected Putts and Diagnostic Reasons

When `accepted` is `false`, the tracker assigns an explanatory `reason`:

| Reason String | Cause | Operator Action |
| :--- | :--- | :--- |
| `"stopped inside the launch window"` | The ball stopped before completing the 300 mm measurement window. | Ensure full stroke; check for debris or mat wrinkles stopping the ball early. |
| `"launch fit too noisy (r2 < 0.98)"` | Distance vs. time linear fit had poor correlation ($r^2 < 0.98$), typically caused by ball hop, skid bounce, or lighting glare. | Ensure flat rolling start; check lighting to eliminate bright glare streaks. |
| `"speed <X> mph out of range"` | Measured speed fell outside the valid bounds (0.3 – 25.0 mph). | Check mat calibration scale; verify ball detection isn't jumping between objects. |
| `"HLA <X> deg out of range"` | Measured horizontal angle exceeded $\pm 45^\circ$. | Verify camera orientation and corner assignment order. |
| `"ball left the view"` | Ball disappeared from camera view before 4 valid points were recorded. | Position resting ball further from the mat edge to allow >= 300 mm travel. |

---

## 7. Real GSPro Connection Verification

- **Port**: GSPro game listens on TCP port `921` (note: `GSPconnect.exe` standalone listens on `1250`; the main game process must be active).
- **Probing**: Run `python -m putting_launch_monitor probe-gspro` to verify connectivity and retrieve player status.
- **Club Code Confirmation**:
  - GSPro sends player state in a code `201` packet:
    ```json
    {"Code": 201, "Message": "GSPro Player Information", "Player": {"Handed": "RH", "Club": "PT"}}
    ```
  - The putter code defaults to `"PT"`. If GSPro configures a different token (e.g. `"P"` or `"PUTTER"`), verify the string in `gspro.py` client settings.
  - Putts are held and not transmitted to GSPro when any club other than a putter is active, preventing accidental rolls from advancing full-shot play.
