FROM python:3.10-slim

RUN mkdir /backend
COPY requirements.txt /backend
WORKDIR /backend

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . /backend

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]