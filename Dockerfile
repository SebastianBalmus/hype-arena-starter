FROM --platform=linux/amd64 python:3.12-slim

ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    ARENA_ARTIFACTS=/app/artifacts

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY arena/ arena/
COPY web/ web/
COPY data/ data/
COPY contracts/ contracts/

RUN python -m arena.train --out /app/artifacts \
    && chmod -R a+rX /app

EXPOSE 8000
CMD ["python", "-m", "arena.cli"]

USER 10001
