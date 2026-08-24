FROM python:3.11-slim

WORKDIR /research
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

