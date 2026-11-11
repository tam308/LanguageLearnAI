#python 3.13 (3.10 no longer gets security fixes). exact version on purpose, bump deliberately
FROM python:3.13.18-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

#requirements first so code changes don't reinstall everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#non-root user with the same uid as the normal user on the server (1000) so it can write the mounted history.json / user_profile.txt
RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin bot && chown bot /app
COPY --chown=bot . .
USER bot

CMD ["python", "languagelearn_AI.py"]
