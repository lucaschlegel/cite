FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# data/ (SQLite, PDFs, Modell-Cache) als Volume einhängen: docker run -v $PWD/data:/app/data -p 8000:8000 -e ANTHROPIC_API_KEY cite
ENV CITE_DB=/app/data/cite.sqlite
EXPOSE 8000
CMD ["uvicorn", "cite.api:app", "--host", "0.0.0.0", "--port", "8000"]
