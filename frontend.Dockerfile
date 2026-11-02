FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /frontend

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group ui

COPY frontend .
COPY .streamlit ./.streamlit

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
