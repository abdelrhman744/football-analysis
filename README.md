# MatchVision — Football Match Analysis Platform

AI-powered football match analysis: YOLOv8 detection, ByteTrack tracking,
heatmaps, team classification, match stats, and an AI chatbot — all in one
FastAPI + Next.js application.

---

## 1. Project Structure

```
football-analysis/
├── frontend/                     # Next.js 14 App Router
│   ├── app/
│   │   ├── page.tsx              # Home / landing
│   │   ├── upload/page.tsx       # Upload + pipeline progress
│   │   ├── dashboard/page.tsx    # Full results dashboard
│   │   └── chat/page.tsx         # AI chatbot
│   ├── components/               # Reusable UI components
│   └── lib/api.ts                # Typed API helpers
│
└── backend/                      # FastAPI
    ├── main.py
    ├── config.py                 ← ALL paths & model settings here
    ├── .env.example              ← copy to .env and fill in keys
    ├── models_weights/           ← PUT YOUR .pt FILE HERE
    │   └── player_ball_detector.pt
    ├── routes/
    │   ├── video.py              # POST /api/video/upload
    │   ├── analysis.py           # POST /api/analysis/start/{id}
    │   │                         # GET  /api/analysis/result/{id}
    │   │                         # GET  /api/analysis/cv-status
    │   ├── chatbot.py            # POST /api/chat
    │   └── match_stats.py        # GET  /api/match-stats/{id}
    ├── services/
    │   ├── cv/                   ← CV sub-package (Kaggle notebook code)
    │   │   ├── __init__.py       # CV_MODEL_AVAILABLE flag
    │   │   ├── detector.py       # YOLOv8 model loader + per-frame inference
    │   │   ├── tracker.py        # ByteTrack via supervision
    │   │   ├── heatmap.py        # Heatmap generation (OpenCV + scipy KDE)
    │   │   ├── video_processor.py # End-to-end pipeline orchestrator
    │   │   └── notebook_adapter.py # Supervision annotators, stats builder
    │   ├── cv_detection_service.py  # Facade → cv/video_processor
    │   ├── tracking_service.py      # Reads tracking.json
    │   ├── heatmap_service.py       # Reads heatmap files
    │   ├── team_classification_service.py
    │   ├── match_stats_service.py
    │   └── chatbot_service.py
    ├── uploads/                  # Raw uploaded videos
    └── results/
        └── {video_id}/
            ├── processed.mp4     # Annotated output video
            ├── detections.json   # Per-frame detections
            ├── tracking.json     # Trajectories
            ├── heatmap_players.png
            ├── heatmap_ball.png
            └── analysis.json     # Full pipeline result (cached)
```

---

## 2. Where to Put Model Weights

```
backend/models_weights/player_ball_detector.pt
```

That path is controlled by `MODEL_WEIGHTS_PATH` in `config.py` (or the env var
`MODEL_WEIGHTS_PATH`).

Your model must be a YOLOv8 model trained on football footage with at least
two classes:
- **Class 0** → player
- **Class 1** → ball

Optionally:
- **Class 2** → referee
- **Class 3** → goalkeeper

Update `CLASS_PLAYER`, `CLASS_BALL`, etc. in `config.py` (or `.env`) to match
your model's `data.yaml`.

### Getting weights from the Kaggle notebook

1. Open the notebook on Kaggle and run all cells.
2. The notebook saves a `best.pt` file in the `runs/detect/train/weights/` directory.
3. Download `best.pt`, rename it to `player_ball_detector.pt`, and place it in
   `backend/models_weights/`.

---

## 3. How to Run the Backend

```bash
cd backend

# 1. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install base deps (no CV yet)
pip install -r requirements.txt

# 3. (When ready) install the CV stack
pip install opencv-python ultralytics supervision numpy scipy torch torchvision

# 4. Copy and fill environment variables
cp .env.example .env
# Edit .env: set MODEL_WEIGHTS_PATH, API keys, etc.

# 5. Start the server
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## 4. How to Run the Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## 5. How to Upload a Video

### Via the web UI
1. Navigate to http://localhost:3000/upload
2. Drag-and-drop or select your `.mp4` / `.mov` / `.avi` / `.mkv` video.
3. Click **Start Analysis**.
4. Watch the pipeline steps complete.
5. Go to the **Dashboard** to see results.

### Via the API directly
```bash
# 1. Upload
curl -X POST http://localhost:8000/api/video/upload \
  -F "file=@match.mp4" | python -m json.tool

# 2. Start analysis (copy video_id from step 1)
curl -X POST http://localhost:8000/api/analysis/start/{video_id}

# 3. Get results
curl http://localhost:8000/api/analysis/result/{video_id}

# 4. Check CV model status
curl http://localhost:8000/api/analysis/cv-status
```

---

## 6. How the CV Pipeline Works

```
upload/ match.mp4
        │
        ▼
cv_detection_service.detect_players_and_ball()
        │
        └──► services/cv/video_processor.process_match_video()
                  │
                  ├── cv2.VideoCapture(video_path)   — open video
                  ├── detector.detect_frame(frame)   — YOLOv8 inference
                  ├── tracker.update(frame_dets)      — ByteTrack assign IDs
                  ├── notebook_adapter.annotate_frame() — draw boxes/labels
                  ├── writer.write(annotated_frame)  — save output video
                  └── heatmap.generate_heatmaps()    — density maps
                  │
                  └──► results/{video_id}/
                            ├── processed.mp4
                            ├── detections.json
                            ├── tracking.json
                            ├── heatmap_players.png
                            └── heatmap_ball.png
```

The pipeline runs in a **single pass** over the video — detect, track, annotate,
and write all happen frame by frame, so even very large videos don't require
loading everything into RAM.

---

## 7. How to Switch from Placeholder to Real Model Mode

| Step | Action |
|------|--------|
| 1    | Install CV libs: `pip install opencv-python ultralytics supervision numpy scipy torch` |
| 2    | Copy your `.pt` file to `backend/models_weights/player_ball_detector.pt` |
| 3    | Update class IDs in `backend/config.py` if they differ from 0/1/2/3 |
| 4    | Restart the backend: `uvicorn main:app --reload` |
| 5    | Visit `http://localhost:8000/api/analysis/cv-status` — should show `"cv_model_available": true` |
| 6    | Upload a video and run analysis — the Dashboard will show the annotated video |

The `GET /api/analysis/cv-status` endpoint always tells you whether the real
model is active. The frontend dashboard also shows a coloured badge.

---

## 8. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/video/upload` | Upload video → `video_id` |
| POST | `/api/analysis/start/{id}` | Run full CV pipeline |
| GET  | `/api/analysis/result/{id}` | Retrieve cached results |
| GET  | `/api/analysis/cv-status` | Check if real model is loaded |
| POST | `/api/chat` | Chat with AI analyst |
| GET  | `/api/match-stats/{match_id}` | External stats (placeholder) |

Static assets served at:
- `/uploads/{filename}` — raw uploaded videos
- `/results/{video_id}/processed.mp4` — annotated video
- `/results/{video_id}/heatmap_players.png` — player heatmap
- `/results/{video_id}/heatmap_ball.png` — ball heatmap
