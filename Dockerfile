# Hype or Truth: the runtime image. Pure Python standard library: no models, no downloads.
FROM python:3.12-slim

WORKDIR /app
COPY hooly/ /app/hooly/
COPY data/practice.csv /app/data/practice.csv

# Readable by the unprivileged runtime user; read-only root at runtime: compile now, never
# write bytecode later; no home directory.
RUN chmod -R a+rX /app && python -m compileall -q /app/hooly
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

USER 10001
CMD ["python", "-m", "hooly.run", "--universe", "/universe", "--output", "/output"]
