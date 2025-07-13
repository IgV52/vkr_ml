FROM debian:stable-slim AS build

ENV BASE_UV_APP=/manager_uv
ENV UV_INSTALL_DIR=${BASE_UV_APP}/bin
ENV UV_PYTHON_INSTALL_DIR=${BASE_UV_APP}/pythons

RUN set -e && \
    apt update && apt install curl -y && \
    curl -LsSf https://github.com/astral-sh/uv/releases/download/0.7.19/uv-installer.sh | sh && \
    $UV_INSTALL_DIR/uv python install 3.13 && \
    rm -rf /var/lib/apt/lists/* && apt clean && \
    set +e

ENV PATH="$UV_INSTALL_DIR:$PATH"

FROM build

WORKDIR /app

COPY ./.python-version ./.python-version
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock

RUN uv sync --no-cache

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

COPY ./cl_okpd2 ./cl_okpd2
COPY ./models ./models
COPY ./manager.py ./manager.py

ENTRYPOINT [ "sh", "-c", "python /app/manager.py predict \"$@\"", "--" ]
