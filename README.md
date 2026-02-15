# KataGo Web

A web-based Go (Weiqi/Baduk) AI interface powered by [KataGo](https://github.com/lightvector/KataGo). Play against one of the strongest Go engines directly from your browser — desktop or mobile.

![Go Board](https://img.shields.io/badge/Game-Go%20%2F%20Weiqi-black) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **KaTrain-style analysis UI** — move candidates with color-coded circles (green → purple gradient based on score difference), win rate bar, and score estimation
- **Multiple play modes** — Free play, play vs AI (Black or White), and AI vs AI
- **Move navigation** — full move history with slider, keyboard shortcuts (← → Home End), and branch navigation
- **Camera board recognition** — take a photo of a real Go board, recognize the position using a CNN deep learning model ([noword/image2sgf](https://github.com/noword/image2sgf)), and continue analysis from there
- **Mobile-friendly** — responsive layout, pinch-to-zoom, two-step move confirmation to prevent misclicks
- **Real stone sounds** — KaTrain-style placement sounds with 5 random variations + capture sound
- **Configurable** — adjustable komi (7.5 / 6.5 / 0.5 / 0), search visits (100–10000), board size (9×9, 13×13, 19×19)

## Architecture

```
katago-web/
├── server/
│   ├── app.py                 # Flask + SocketIO web server
│   ├── katago_engine.py       # KataGo process manager (Analysis JSON API)
│   └── noword_recognizer.py   # CNN board recognition (FCOS + EfficientNet)
├── static/
│   ├── index.html             # Main UI
│   ├── css/style.css          # Responsive styles
│   ├── js/
│   │   ├── app.js             # Application logic, WebSocket communication
│   │   └── goboard.js         # Canvas board rendering, interaction, sounds
│   └── sounds/                # Stone placement & capture audio files
├── config/
│   └── default_gtp.cfg        # KataGo engine configuration
├── models/
│   └── image2sgf/             # CNN model weights (board.pth + stone.pth)
├── setup.ps1                  # One-click Windows setup script
└── requirements.txt
```

## Prerequisites

- **Python 3.10+**
- **KataGo** — download from [KataGo releases](https://github.com/lightvector/KataGo/releases) (OpenCL or CUDA backend)
- **KataGo model weights** — download from [KataGo models](https://katagotraining.org/)
- **NVIDIA GPU** (recommended) — for both KataGo inference and CNN board recognition

## Installation

### Quick Setup (Windows)

Run the automated setup script:

```powershell
.\setup.ps1
```

This will download KataGo, model weights, install Python dependencies, generate tuning, and start the server.

### Manual Setup

1. **Install KataGo** somewhere on your system (e.g., `C:\katago\`):
   - `katago.exe` (OpenCL or CUDA build)
   - A model weights file (`.bin.gz`)

2. **Install Python dependencies**:

   ```bash
   pip install flask flask-socketio flask-cors eventlet opencv-python-headless sgfmill
   ```

3. **Install PyTorch** (for camera board recognition):

   ```bash
   # CUDA (recommended, much faster)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

   # CPU only
   pip install torch torchvision
   ```

4. **Download CNN models** for board recognition (optional):

   Download `board.pth` and `stone.pth` from [noword/image2sgf](https://github.com/noword/image2sgf) and place them in `models/image2sgf/`.

5. **Configure paths** in `server/app.py` or via environment variables:

   ```bash
   export KATAGO_PATH=/path/to/katago
   export KATAGO_MODEL=/path/to/model.bin.gz
   ```

## Usage

```bash
cd katago-web
python server/app.py
```

Open `http://localhost:5000` in your browser.

### Remote Access

To access from your phone or another device, use [Tailscale](https://tailscale.com/) or any VPN/tunneling solution, then visit `http://<your-ip>:5000`.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` | Back 1 move |
| `→` | Forward 1 move |
| `Home` | Jump to start |
| `End` | Jump to latest |
| `Ctrl+←` | Back 10 moves |
| `Ctrl+→` | Forward 10 moves |

## Configuration

### KataGo Engine

Edit `config/default_gtp.cfg` to tune KataGo parameters:

- `numSearchThreads` — number of search threads (default: 16)
- Search visits are controlled from the web UI (100–10000, default: 3000)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KATAGO_PATH` | `C:\katago\katago.exe` | Path to KataGo executable |
| `KATAGO_MODEL` | `C:\katago\kata1-b18c384nbt-*.bin.gz` | Path to model weights |
| `KATAGO_CONFIG` | `config/default_gtp.cfg` | Path to KataGo config |
| `PORT` | `5000` | Server port |
| `DEFAULT_MAX_VISITS` | `3000` | Default analysis visits |

## How It Works

1. **Backend**: Flask + SocketIO server manages a KataGo subprocess via the Analysis JSON API
2. **Frontend**: Canvas-based Go board with real-time WebSocket updates
3. **Analysis**: KataGo returns top move candidates with win rates, scores, and principal variations — rendered as KaTrain-style colored circles
4. **Recognition**: Photos of real boards are processed by a two-stage CNN pipeline:
   - **FCOS** (ResNet50-FPN) detects the four corners of the board
   - **EfficientNet-B3** classifies each intersection as empty / black / white

## Credits

- [KataGo](https://github.com/lightvector/KataGo) by lightvector — the Go engine
- [noword/image2sgf](https://github.com/noword/image2sgf) — CNN board recognition models
- [KaTrain](https://github.com/sanderland/katrain) — UI design inspiration and stone sounds

## License

MIT
