# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    DEEPFAKE DETECTOR - TERMINAL UI                           ║
# ║                              README                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

```
██████╗ ███████╗███████╗██████╗ ███████╗ █████╗ ██╗  ██╗███████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝█████╗  ███████║█████╔╝ █████╗  
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  
██████╔╝███████╗███████╗██║     ██║     ██║  ██║██║  ██╗███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

## SYSTEM REQUIREMENTS

```
> NODE.JS:     v18.0.0 or higher
> PYTHON:      v3.10 or higher
> RAM:         8GB minimum
> GPU:         Optional (CPU inference supported)
```

## QUICK START

### Option 1: Automated Launch

```bash
chmod +x start.sh
./start.sh
```

### Option 2: Manual Launch

#### Terminal 1 - Python API Server
```bash
cd /path/to/FINAL\ DEEPFAKE\ PROJECT\ MODEL
python -m uvicorn deepfake-terminal-ui.api.server:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2 - Next.js Frontend
```bash
cd deepfake-terminal-ui
npm install
npm run dev
```

### Access Points

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:3000      |
| API      | http://localhost:8000      |
| API Docs | http://localhost:8000/docs |

## FEATURES

```
[✓] IMAGE ANALYSIS
    > Upload any image (JPG, PNG, WebP)
    > Xception neural network inference
    > Grad-CAM heatmap visualization
    > Facial region attribution analysis
    > Confidence scoring

[✓] VIDEO ANALYSIS  
    > Upload videos (MP4, MOV, AVI)
    > Frame-by-frame analysis
    > Suspicion timeline visualization
    > Multiple verdict methods (Mean, Max, Majority)

[✓] TERMINAL CLI AESTHETIC
    > Monospace typography
    > Phosphor glow effects
    > CRT scanline overlay
    > ASCII art elements
    > Brutalist design language
```

## PROJECT STRUCTURE

```
deepfake-terminal-ui/
├── api/
│   ├── server.py          # FastAPI backend
│   └── requirements.txt   # Python dependencies
├── src/
│   ├── app/
│   │   ├── layout.tsx     # Root layout with CRT effects
│   │   ├── page.tsx       # Main detection page
│   │   └── globals.css    # Terminal design tokens
│   ├── components/
│   │   ├── TerminalWindow.tsx
│   │   ├── TerminalButton.tsx
│   │   ├── TerminalProgress.tsx
│   │   ├── FileUpload.tsx
│   │   ├── FacialAttribution.tsx
│   │   └── AsciiArt.tsx
│   └── lib/
│       ├── api.ts         # API client
│       └── types.ts       # TypeScript interfaces
├── package.json
├── tailwind.config.ts
├── start.sh               # Launch script
└── README.md
```

## DESIGN SYSTEM

### Colors
```
BACKGROUND:  #0a0a0a  (Deep black)
PRIMARY:     #33ff00  (Terminal green)
SECONDARY:   #ffb000  (Amber warning)
MUTED:       #1f521f  (Dimmed green)
ERROR:       #ff3333  (Bright red)
```

### Typography
```
FONT:        JetBrains Mono, Fira Code, VT323
STYLE:       UPPERCASE for headers
GLOW:        text-shadow: 0 0 5px rgba(51, 255, 0, 0.5)
```

### Components
```
BUTTONS:     [ LABEL ] with inverted hover
WINDOWS:     ┌── TITLE ──┐ with status badge
PROGRESS:    [████████░░░░] 65.5%
STATUS:      [OK] [ERR] [WARN] [...]
```

## API ENDPOINTS

### `GET /health`
Health check endpoint.

### `POST /analyze/image`
Analyze an image for deepfake detection.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `include_gradcam`: Include Grad-CAM heatmap (default: true)
- `include_regions`: Include facial region analysis (default: true)

### `POST /analyze/video`
Analyze a video for deepfake detection.

**Parameters:**
- `file`: Video file (multipart/form-data)
- `sample_every`: Sample every N frames (default: 15)
- `max_frames`: Maximum frames to analyze (default: 40)
- `method`: Verdict method - mean, max, majority (default: mean)

## TROUBLESHOOTING

### API Connection Error
```
[ERR] Make sure the Python API is running on port 8000
$ python -m uvicorn deepfake-terminal-ui.api.server:app --port 8000
```

### Model Loading Error
```
[ERR] Ensure weights/xception_gan_augmented.pth exists
[>] Check the checkpoint path in api/server.py
```

### Port Already in Use
```
$ lsof -i :3000
$ lsof -i :8000
$ kill -9 <PID>
```

---

```
╔════════════════════════════════════════════════════════════════════╗
║  DEEPFAKE DETECTOR v1.0.0                                          ║
║  Built with: Next.js + FastAPI + PyTorch + Xception                ║
║  Design: Terminal CLI Aesthetic                                    ║
╚════════════════════════════════════════════════════════════════════╝
```
