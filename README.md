# PYiOS 📱⚡

> A lightweight Python-based ARMv7 iOS Binary Emulator and Classic Simulator designed to run legacy iPhone OS applications directly on your desktop.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)

**PYiOS** is an experimental compatibility layer and emulator that runs classic iOS binaries. Powered by the **Unicorn Engine** for ARMv7 emulation, **Pygame**, and **PyOpenGL** for rendering, it bridges low-level system calls, implements a lightweight Objective-C runtime, and simulates foundational iOS frameworks (like UIKit, Foundation, and CoreGraphics) to bring vintage apps back to life.

---

## ✨ Features

* **ARMv7 CPU Emulation:** Powered by Unicorn CPU emulation to execute native compiled iOS binary code.
* **Objective-C Runtime Bridge:** Dynamic method dispatching, class registration, and messaging shims tailored for early iOS binaries.
* **Accelerated Graphics Pipeline:** Software Quartz context rendering coupled with a PowerVR SGX compatibility layer via PyOpenGL and Pygame.
* **Classic Home Screen UI:** Features an interactive classic iOS home screen with IPA file extraction, app icon masking, badge support, and smooth pagination.
* **Specialized Asset Decoders:** Built-in decoders for Apple's optimized `CgBI` PNG graphic format and Core Audio Format (`.caf`) sound files.
* **Sensor Simulation:** Built-in accelerometer mapping for tilt, motion, and shaking actions via keyboard shortcuts.
* **Modular Architecture:** Cleanly separated subsystems for CPU, GPU, audio, sensors, and UI animations.

---

## 🛠️ Project Structure

```text
├── main.py            # Main entry point and lifecycle controller
├── homescreen.py      # Classic iOS springboard interface & IPA extractor
├── cpu.py             # ARMv7 CPU execution loop, Obj-C runtime & system shims
├── gpu.py             # PowerVR SGX graphics compatibility pipeline
├── coregraphics.py    # Software Quartz 2D context rendering
├── bridge.py          # Objective-C selector bridge and NIB parser
├── animations.py      # UIKit and Core Animation easing/tweening engine
├── accelerometer.py   # Motion, tilt, and shake hardware sensor simulation
├── audio.py           # System sound player and audio manager
├── caf_audio.py       # Apple Core Audio Format (.caf) decoder
└── cgbi_png.py        # Apple-optimized CgBI PNG graphics loader
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pygame PyOpenGL unicorn
```

### Running the Emulator

Clone the repository and launch the main entry point to open the classic iOS home screen:

```bash
python main.py
```

---

## 🎮 Controls & Interaction

* **Mouse / Touch:** Click and drag to interact with app icons, sliders, and UI components.
* **Accelerometer Simulation:** Use keyboard shortcuts (configured in `accelerometer.py`) to simulate device tilt, rotation, and shaking gestures (e.g., shaking *iBeer*).

---

## 🤝 Contributing

Contributions, bug reports, and pull requests are welcome! Feel free to open an issue if you encounter unhandled ARM instructions, missing framework symbols, or graphics rendering bugs.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
