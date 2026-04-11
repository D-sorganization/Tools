# Chaotic Double Pendulum Screensaver

This is a visually striking, continuous simulation of a double pendulum evaluated using pristine Lagrangian mechanic differential equations. It's intended to be utilized either strictly for its video-export capabilities, or just run completely live as a desktop decoration/screensaver.

## Features

- **Accurate Lagrangian Engine**: Solved flawlessly and optimized using `scipy.integrate.solve_ivp` RK45 methods.
- **Cool Aesthetics**: Dark theme layout styling with dynamic trailing particle effects and `Ellipse` shapes mapping their rotational orientation precisely to rod tangents.
- **Video Export Mode**: Generate raw `.mp4` loop files of the simulation for distribution!

## Setup Let's Go

First, ensure your environment exists. Best practice is to construct a venv:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

_(Optional)_ If you plan to export the simulation natively to video out of Matplotlib using the `--save` hook, ensure you have **FFmpeg** installed and hooked onto your Windows `PATH`.

## How to execute

### 1) Endless Live Display (Standard Screensaver)

Spawns the live matplotlib window natively and calculates on your machine live.

```bash
python chaotic_pendulum.py
```

### 2) Video Generation Mode

Calculates Lagrangian constraints in the background entirely, writing frame-by-frame out to a finished `.mp4`.

```bash
python chaotic_pendulum.py --save "my_screensaver.mp4" --duration 60 --fps 60
```

### Advanced Config

You can also tweak variables like inner/outer masses or pendulum lengths directly via CLI.

```bash
python chaotic_pendulum.py --m1 2.5 --l2 0.8
```

See `-h` for all options.
