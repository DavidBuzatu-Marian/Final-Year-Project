FROM python:3.9

WORKDIR /app

RUN wget --quiet https://releases.hashicorp.com/terraform/1.1.0/terraform_1.1.0_linux_amd64.zip \
    && unzip terraform_1.1.0_linux_amd64.zip \
    && mv terraform /usr/bin \
    && rm terraform_1.1.0_linux_amd64.zip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN cd ./terraform && terraform init && cd ..

ENV FLASK_ENV=development

ENV FLASK_APP=app.py
# Host is chosen as 0.0.0.0 to make it available outside of container
CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0" ]