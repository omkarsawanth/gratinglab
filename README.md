# GratingLab

**Design and Development of a Simulation to Study Laser Diffraction Patterns from Optical Gratings and Its Utility in Advanced Material Characterization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular wave-optics simulation and inverse material characterization suite created for university physics micro-projects. GratingLab bridges theoretical electromagnetism, numerical wave optics (Fraunhofer FFT & Angular Spectrum Method), and real-world industrial optical metrology (semiconductor wafer pitch estimation, optical disc diagnostics, and thermal strain sensing).

---

## 1. Installation

### System Requirements
- Python 3.10 or newer
- Minimal dependencies: NumPy, SciPy, Matplotlib, pytest

```bash
git clone https://github.com/omkarsawanth/gratinglab.git
cd gratinglab

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Quickstart & CLI Commands

```bash
# Launch the Interactive Optical Studio:
python main.py

# Launch with custom experimental parameters:
python main.py --grating binary --wavelength 532 --period 0.74 --slits 20 --screen-dist 0.8

# Run all three material characterization case studies:
python -m gratinglab.case_studies

# Run optical bench CD/DVD validation report:
python validation/cd_dvd_validation.py

# Run comprehensive test suite with coverage:
pytest tests/ --cov=gratinglab --cov-report=term-missing
```

---

## 3. Graphical User Interface (GUI) Manual

When executing `python main.py`, the interactive studio opens:

```text
+---------------------------------------------------------------------------------------+
|  Grating Transmittance t(x)              |  Far-Field Diffraction Intensity I(θ)      |
|  - Amplitude |t(x)| (blue)               |  - Numerical FFT (green)                   |
|  - Phase φ(x) / π (orange dashed)        |  - Analytical N-slit model (red dotted)    |
|                                          |  - Analytical order markers (m = 0, ±1...) |
+---------------------------------------------------------------------------------------+
|  Simulated Physical Observation Screen Strip at Distance L                            |
|  (Renders realistic laser illumination using exact wavelength-matched RGB colormap)   |
+---------------------------------------------------------------------------------------+
|  [Radio] Grating Profile   |  Sliders (Column 1):         |  Sliders (Column 2):      |
|   (•) Binary Amplitude     |  - Wavelength: 400..700 nm   |  - Incidence θᵢ: ±30°     |
|   ( ) Sinusoidal Phase     |  - Period d: 0.5..20.0 µm    |  - Screen L: 0.1..2.0 m   |
|   ( ) Blazed Sawtooth      |  - Duty Cycle: 0.05..0.95    |  - Beam Waist w₀: 5..100µm|
|   ( ) 2D Orthogonal        |  - Phase Depth: 0.1..4π rad  |  - [Button] Talbot Carpet |
|  [Checks]                  |  - Slit Count N: 2..50       |---------------------------|
|   [x] Log Intensity Scale  +------------------------------+  [Text Info Box]          |
|   [x] Compare Analytical   | Summary of propagating orders, angles θₘ, missing orders |
+---------------------------------------------------------------------------------------+
```

### Slider Controls Reference
| Parameter | Slider Name | Physical Meaning & Units | Default |
| :--- | :--- | :--- | :--- |
| $\lambda$ | **Wavelength** | Laser source wavelength in nanometers [nm]. Automatically updates the screen strip colormap to match human color vision. | 650.0 nm |
| $d$ | **Period d** | Grating pitch / ruling spacing in micrometers [$\mu\text{m}$]. Controls diffraction order angular separation. | 1.60 $\mu\text{m}$ |
| $\gamma$ | **Duty / Slit frac** | Ratio of transparent slit width to period ($a / d$). Governs the single-slit envelope zeros and missing orders. | 0.50 |
| $\phi_0$ | **Phase Depth** | Peak-to-peak phase modulation in radians for sinusoidal and blazed phase profiles. | $2\pi$ rad |
| $N$ | **Slits / Lines N** | Number of illuminated slits. Governs the sharpness (FWHM $\propto 1/N$) and secondary maxima count ($N-2$). | 10 |
| $\theta_i$ | **Incidence $\theta_i$** | Laser beam tilt angle in degrees [$^\circ$] relative to normal incidence. Introduces an asymmetric phase tilt. | $0.0^\circ$ |
| $L$ | **Screen Dist L** | Distance from grating aperture to observation screen in meters [m]. Scales physical spot offset $y = L \tan\theta$. | 0.50 m |
| $w_0$ | **Beam Waist** | Gaussian electric field beam waist radius in micrometers [$\mu\text{m}$]. Simulates realistic laser beam profiles. | 40.0 $\mu\text{m}$ |

---

## 4. Theoretical Physics Model

### 1. The Generalized Grating Equation
For an optical grating with pitch $d$, incident at angle $\theta_i$, the $m$-th diffraction order angle $\theta_m$ satisfies:
$$d \left(\sin\theta_m - \sin\theta_i\right) = m \lambda$$

Propagating orders exist strictly when:
$$\left|\sin\theta_i + \frac{m\lambda}{d}\right| \le 1$$
Orders with $|\sin\theta_m| > 1$ are **evanescent**; their field decays exponentially within a distance of $\sim \lambda$ from the grating surface.

### 2. Fraunhofer $N$-Slit Diffraction Intensity
The normalized far-field intensity distribution from $N$ identical slits of width $a$ and pitch $d$ is:
$$I(\theta) = \left[\frac{\sin\left(\frac{\pi a}{\lambda}\left(\sin\theta - \sin\theta_i\right)\right)}{\frac{\pi a}{\lambda}\left(\sin\theta - \sin\theta_i\right)}\right]^2 \times \left[\frac{\sin\left(N \frac{\pi d}{\lambda}\left(\sin\theta - \sin\theta_i\right)\right)}{N \sin\left(\frac{\pi d}{\lambda}\left(\sin\theta - \sin\theta_i\right)\right)}\right]^2$$

When $\delta = \frac{\pi d}{\lambda}(\sin\theta - \sin\theta_i) \to m\pi$, the multi-slit interference term approaches $1.0$ by L'Hôpital's rule.

### 3. Missing Orders
When a principal interference maximum coincides with a zero of the single-slit diffraction envelope:
$$m = p \left(\frac{d}{a}\right) \quad (p \in \mathbb{Z} \setminus \{0\})$$
For a $50\%$ duty cycle ($d/a = 2$), all even orders ($m = \pm 2, \pm 4, \dots$) vanish.

### 4. Wave Optics (Fraunhofer FFT & Angular Spectrum Method)
- **Far Field (Fraunhofer)**:
  $$\mathcal{F}\{ U(x) \}_{f_x} \implies \sin\theta = \lambda f_x$$
- **Near-to-Intermediate Field (Angular Spectrum)**:
  $$U(x, z) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{U(x, 0)\} \cdot \exp\left(i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2}\right) \right\}$$
- **Talbot Self-Imaging**:
  At integer multiples of the Talbot distance $z_T = \frac{2d^2}{\lambda}$, periodic wavefields undergo complete constructive re-phasing, replicating the grating transmission profile without lenses.

---

## 5. Physical Assumptions & Validity Regime

### 1. Scalar Wave Approximation ($d \gg \lambda$)
GratingLab operates under the **Scalar Theory of Diffraction**. This assumes:
- The electromagnetic field can be described by a single scalar wave amplitude $U(\mathbf{r})$.
- Vector boundary conditions (polarization decoupling, surface plasmon resonances, and TE/TM polarization splitting) are negligible.

> [!IMPORTANT]
> **Scalar Validity Domain**:
> Scalar theory is quantitatively accurate when the grating pitch $d \ge 3\lambda$. For a red laser ($\lambda = 650\,\text{nm}$), scalar theory is valid for $d \ge 2.0\,\mu\text{m}$.
> For standard CDs ($d = 1.6\,\mu\text{m} \approx 2.46\lambda$) and DVDs ($d = 0.74\,\mu\text{m} \approx 1.14\lambda$), the grating equation correctly predicts angular spot positions; however, polarization effects cause minor deviations in relative peak efficiencies.

### 2. Future Work: Vector Electromagnetics (RCWA)
When $d \sim \lambda$ or $d < \lambda$ (e.g. Blu-ray discs with $d = 320\,\text{nm}$), full Maxwell equations must be solved. Future extensions of GratingLab will incorporate **Rigorous Coupled-Wave Analysis (RCWA)** to account for:
- Transverse Electric (TE) vs Transverse Magnetic (TM) polarization efficiencies.
- Finite metallic conductivity and complex refractive indices $\tilde{n} = n + ik$.
- Deep groove aspect ratios and Wood's anomalies.

---

## 6. Validation

### (a) Analytical Validation
The numerical wave propagation engine (`propagation.far_field_fft`) was benchmarked against the closed-form analytical grating equation ($d \sin\theta_m = m\lambda$) for standard optical media at $\lambda = 650\,\text{nm}$ (normal incidence $\theta_i = 0^\circ$).

Grid parameters: $W = 100 \times d$, $N = 8192$ samples, zero-padding factor $P = 4$ ($M = 32768$).

| Media Track Standard | Order ($m$) | Grating Equation $\theta_m$ | Simulator FFT $\theta_m$ | Absolute Difference | Percentage Difference | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Compact Disc (CD)**<br>($d = 1.600\,\mu\text{m}$) | $0$ | $0.0000^\circ$ | $0.0000^\circ$ | $0.0000^\circ$ | **0.000%** | Exact match |
| | $+1$ | $23.9763^\circ$ | $23.9763^\circ$ | $< 0.0001^\circ$ | **0.000%** | Exact match |
| | $+2$ | $54.3392^\circ$ | $54.3392^\circ$ | $< 0.0001^\circ$ | **0.000%** | Exact match |
| | $+3$ | Evanescent | Evanescent | — | — | Correctly cut off ($|\sin\theta| > 1$) |
| **Digital Versatile Disc (DVD)**<br>($d = 0.740\,\mu\text{m}$) | $0$ | $0.0000^\circ$ | $0.0000^\circ$ | $0.0000^\circ$ | **0.000%** | Exact match |
| | $+1$ | $61.4343^\circ$ | $61.4343^\circ$ | $< 0.0001^\circ$ | **0.000%** | Exact match |
| | $+2$ | Evanescent | Evanescent | — | — | Correctly cut off ($|\sin\theta| > 1$) |

> **Conclusion**: The numerical FFT spectral sampling aligns with the analytical solution with **0.000% error** because the spatial window $W_{\text{pad}} = P \cdot W = 400 d$ forms an exact harmonic integer grid for the spatial frequencies $f_x = m / d$.

---

### (b) Experimental Validation

> **Status**: Pending - laser pointer + CD/DVD experiment scheduled for Nov 2026.

#### Experimental Procedure (Reflection Geometry at Normal Incidence)
1. **Apparatus Setup**:
   - Mount a commercial Compact Disc (CD-R) or Digital Versatile Disc (DVD-R) vertically on an optical rail holder.
   - Position a calibrated laser diode pointer ($\lambda = 650\,\text{nm}$ red diode or $\lambda = 532\,\text{nm}$ DPSS green) at normal incidence ($\theta_i = 0^\circ$) to the disc recording tracks.
   - Place a flat, diffuse white observation screen parallel to the disc face at a measured perpendicular distance $L$ (recommended $L = 0.50\,\text{m}$ to $1.00\,\text{m}$).
2. **Measurement Protocol**:
   - Record distance $L$ from the disc track plane to the observation screen using a precision steel rule ($\pm 0.5\,\text{mm}$).
   - With the room dimmed, mark the central specular reflection spot ($m = 0$) and the symmetric positive and negative diffraction orders ($m = \pm 1, \pm 2$).
   - Measure the linear transverse displacement $y_m$ from the center spot to each order.
   - Compute the experimental diffraction angle:
     $$\theta_m = \arctan\left(\frac{y_m}{L}\right)$$
   - Calculate track pitch using the grating equation:
     $$d_{\text{meas}} = \frac{m \lambda}{\sin\theta_m}$$
   - Repeat over 3 distinct screen distances ($L = 0.5\,\text{m}, 0.75\,\text{m}, 1.0\,\text{m}$) to eliminate zero-offset systematic bias.
3. **Safety Protocol**:
   - Class 2 / 3R laser safety rules apply.
   - **Never look directly into the primary laser aperture or specular reflected beams.**
   - Restrict beam paths to bench level and terminate stray reflections with non-reflective beam stops.


---

## 7. Performance Benchmarks & Test Report

### Runtime Benchmarks (100 repetitions per module)
Measured with `timeit` on Python 3.11:

| Module & Function | Benchmark Scenario | Measured Mean Latency |
| :--- | :--- | :---: |
| `gratinglab.constants.nm_to_m` | Scalar float conversion | **0.14 µs** |
| `gratinglab.analytical.order_angles` | $d = 1.6\,\mu\text{m}$, $m_{\max}=4$ | **2.36 µs** |
| `gratinglab.analytical.missing_orders` | $d = 2.0\,\mu\text{m}$, $a = 1.0\,\mu\text{m}$ | **2.58 µs** |
| `gratinglab.analytical.talbot_distance` | $d = 1.6\,\mu\text{m}$, $\lambda = 650\,\text{nm}$ | **0.21 µs** |
| `gratinglab.analytical.n_slit_intensity` | $N = 20$, 2048 angular points | **76.92 µs** |
| `gratinglab.gratings.binary_amplitude` | 2048 grid samples | **35.10 µs** |
| `gratinglab.gratings.sinusoidal_phase` | 2048 grid samples | **62.46 µs** |
| `gratinglab.gratings.blazed` | 2048 grid samples | **64.16 µs** |
| `gratinglab.propagation.gaussian_beam` | 2048 grid samples | **12.04 µs** |
| `gratinglab.propagation.far_field_fft` | $N = 2048$, pad factor = 2 ($M = 4096$) | **90.17 µs** |
| `gratinglab.propagation.angular_spectrum` | $N = 2048$, single $z$-step | **89.80 µs** |
| `gratinglab.characterize.find_orders` | 2048 sample points | **25.18 µs** |
| `gratinglab.characterize.estimate_period` | Linear regression across 5 orders | **179.99 µs** |

### Automated Testing & Coverage Status
The complete test suite runs via pytest:
```bash
pytest tests/ --cov=gratinglab --cov-report=term
```
**Results (`docs/test_report.txt`):**
- **34 of 34 unit tests passing (100% pass rate)**.
- Validates analytical limits, missing order cancellations, edge cases (deep sub-wavelength cutoffs $d < \lambda$, grazing angles), Parseval energy conservation, and inverse material characterizations.
