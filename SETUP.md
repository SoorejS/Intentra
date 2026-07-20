# Intentra v2 - Setup Instructions

## Folder Structure

Put everything like this inside your Intentra folder:

```
Intentra/
├── index.html          (your existing frontend)
├── style.css           (your existing styles)
├── app.js              (REPLACE with app_connected.js content)
├── main.py             (NEW - FastAPI backend)
├── requirements.txt    (NEW - Python dependencies)
├── core/
│   ├── __init__.py     (create empty file)
│   ├── intent_schema.py
│   ├── dataset_generator.py
│   └── evaluator.py
```

---

## Step 1: Create the core folder and __init__.py

In your Intentra folder, create a folder called `core`.
Inside it, create an empty file called `__init__.py`.

---

## Step 2: Set your Anthropic API key

Windows (Command Prompt):
```
set ANTHROPIC_API_KEY=your_key_here
```

Windows (PowerShell):
```
$env:ANTHROPIC_API_KEY="your_key_here"
```

Get your key from: https://console.anthropic.com

---

## Step 3: Install Python dependencies

```
pip install anthropic fastapi uvicorn pydantic python-multipart
```

---

## Step 4: Run the backend

```
cd Intentra
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

Test it works:
Open http://localhost:8000 in browser - should show {"status": "Intentra API running"}

---

## Step 5: Connect the frontend

In your index.html, replace the script tag at the bottom:
```html
<script src="app.js"></script>
```
with:
```html
<script src="app_connected.js"></script>
```

---

## Step 6: Open the frontend

Open index.html in your browser (or run python -m http.server 8080 in the Intentra folder).

Type an objective and click Generate Dataset.

The frontend will call your real FastAPI backend which calls the Anthropic API and returns a real dataset.

---

## Troubleshooting

**CORS error in browser console:**
The backend already has CORS enabled for all origins. Make sure uvicorn is running.

**API key error:**
Make sure ANTHROPIC_API_KEY is set in the same terminal where you run uvicorn.

**JSON parse error:**
The LLM occasionally returns malformed JSON. The pipeline will retry. If it keeps failing, try a more specific objective.

**Module not found:**
Make sure you created the core/__init__.py file and all four core Python files are in the core/ folder.

---

## Testing the API directly

Once running, test with curl:

```
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"objective\": \"detect urgent customer complaints\", \"dataset_size\": 10}"
```

Or open http://localhost:8000/docs for the automatic Swagger UI.
