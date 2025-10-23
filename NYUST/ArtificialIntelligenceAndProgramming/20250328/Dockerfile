FROM python:3.12

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]