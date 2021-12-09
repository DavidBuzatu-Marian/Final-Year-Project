FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENV FLASK_ENV=development

ENV FLASK_APP=app.py
# Host is chosen as 0.0.0.0 to make it available outside of container
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0" ]