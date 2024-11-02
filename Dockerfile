FROM python:3.12-slim

RUN pip install poetry==1.8.3

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY ./app ./app
COPY ./vector_database ./vector_database
COPY .env /code/.env

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
