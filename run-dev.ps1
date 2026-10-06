# Starts the FastAPI backend with UTF-8 forced (DeepFace's logger emits emoji
# that the default Windows cp1252 console cannot encode).
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn backend.app.main:app --reload --port 8000
