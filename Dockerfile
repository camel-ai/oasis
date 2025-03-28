FROM python:3.11

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir --upgrade -e .


CMD ["python", "scripts/base/listen.py"]