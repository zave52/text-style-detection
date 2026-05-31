# API Reference

The backend exposes a RESTful API built with **FastAPI**. All endpoints accept and return JSON.

When running via Docker Compose, the API is available at `http://localhost:8000`. Interactive documentation (Swagger UI) is available at `http://localhost:8000/docs`.

---

## Endpoints

### `POST /predict`

Predict both **style** and **tone** for the given text.

=== "Request"

    ```json
    {
      "text": "Your text here"
    }
    ```

=== "Response (200)"

    ```json
    {
      "style": "academic",
      "tone": "neutral"
    }
    ```

=== "cURL"

    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text here"}'
    ```

---

### `POST /predict/style`

Predict only the **style** of the given text.

=== "Request"

    ```json
    {
      "text": "Your text here"
    }
    ```

=== "Response (200)"

    ```json
    {
      "style": "formal"
    }
    ```

=== "cURL"

    ```bash
    curl -X POST http://localhost:8000/predict/style \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text here"}'
    ```

---

### `POST /predict/tone`

Predict only the **tone** of the given text.

=== "Request"

    ```json
    {
      "text": "Your text here"
    }
    ```

=== "Response (200)"

    ```json
    {
      "tone": "sarcastic"
    }
    ```

=== "cURL"

    ```bash
    curl -X POST http://localhost:8000/predict/tone \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text here"}'
    ```

---

## Schemas

### Request

#### TextRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | Yes | The input text to classify |

### Responses

#### PredictResponse (`/predict`)

| Field | Type | Description |
|-------|------|-------------|
| `style` | `string` | Predicted writing style |
| `tone` | `string` | Predicted emotional tone |

#### StyleResponse (`/predict/style`)

| Field | Type | Description |
|-------|------|-------------|
| `style` | `string` | Predicted writing style |

#### ToneResponse (`/predict/tone`)

| Field | Type | Description |
|-------|------|-------------|
| `tone` | `string` | Predicted emotional tone |

---

## Possible Values

### Style

| Value | Description |
|-------|-------------|
| `academic` | Scholarly, research-oriented writing |
| `business` | Professional corporate communication |
| `formal` | Official language following conventions |
| `informal` | Casual, conversational writing |
| `literary` | Creative, artistic prose |

### Tone

| Value | Description |
|-------|-------------|
| `aggressive` | Hostile, confrontational language |
| `friendly` | Warm, supportive communication |
| `neutral` | Objective, emotionally detached |
| `sarcastic` | Ironic, meaning differs from literal |
| `urgent` | Time-sensitive, action-demanding |

---

## Error Responses

### 503 — Model Not Loaded

Returned when the ML models haven't finished loading (e.g., right after startup).

```json
{
  "detail": "Style model is not loaded."
}
```

### 503 — NLP Model Not Loaded

Returned when the spaCy NLP pipeline is unavailable.

```json
{
  "detail": "NLP model is not loaded."
}
```

### 422 — Validation Error

Returned when the request body is missing or malformed.

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
