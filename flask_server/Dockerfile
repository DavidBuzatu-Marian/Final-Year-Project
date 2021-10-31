FROM python:3.7-alpine

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

# Host is chosen as 0.0.0.0 to make it available outside of container
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0" ]