# Chapter 9 — Media Processing

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 9.1 Video Processor

**Source:** `src/media_processing/video_processor/`
**Status:** ✅ Implemented (Complex multi-platform application)

### 9.1.1 Purpose

Comprehensive video processing platform with web application, scientific auditing capabilities, and golf swing analysis.

### 9.1.2 Architecture

| Component | Technology | Path |
|-----------|-----------|------|
| Web App | Next.js / React / TypeScript | `apps/web/` |
| Core Processing | Python | `python/video_processor_src/` |
| Scientific Auditor | Python | `tools/scientific_auditor.py` |
| Golf Swing Analyzer | TypeScript | `apps/web/lib/golf/swingAnalyzer.ts` |
| MATLAB Models | MATLAB | `matlab/models/` |

### 9.1.3 Features

- Video upload and processing pipeline
- Frame extraction and analysis
- Golf swing biomechanics analysis
- Scientific video auditing
- Motion tracking
- Web-based platform with Next.js frontend
- Input sanitization (`apps/web/lib/sanitize.ts`)
- Structured logging (`apps/web/lib/logger.ts`)

### 9.1.4 Golf Swing Analysis

The swing analyzer applies biomechanical analysis including:

- Joint angle tracking
- Swing plane analysis
- Tempo and timing metrics
- Club head speed estimation

### 9.1.5 MATLAB Integration

- Pendulum model simulation (`matlab/models/pendulum_model.m`)
- Physics-based motion modeling

---

## 9.2 Audio Processor

**Source:** `src/media_processing/audio_processor/`
**Status:** ✅ Implemented (MATLAB)

### 9.2.1 Purpose

Audio signal processing tool implemented primarily in MATLAB.

### 9.2.2 Capabilities

- Audio file import/export
- Signal processing (filtering, FFT analysis)
- Spectral analysis and visualization
- Audio feature extraction

### 9.2.3 Key Equations

**Fast Fourier Transform (FFT):**

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}$$

**Spectrogram (Short-Time Fourier Transform):**

$$S(t, f) = \left| \int_{-\infty}^{\infty} x(\tau) \cdot w(\tau - t) \cdot e^{-j 2\pi f \tau} \, d\tau \right|^2$$

where $w$ is the window function.

**Power Spectral Density:**

$$PSD(f) = \frac{|X(f)|^2}{N \cdot f_s}$$

---

*[← Web Applications](./08_web_applications.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Development Tools →](./10_development_tools.md)*
