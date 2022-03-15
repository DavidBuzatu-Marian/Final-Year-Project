#! /bin/bash

sudo systemctl restart systemd-networkd.service
sudo apt update
sudo apt -y install docker.io
sudo gcloud auth configure-docker -q
sudo docker pull eu.gcr.io/finalyearproject-338819/fyp_instance:latest
sudo docker run -p 5000:5000 eu.gcr.io/finalyearproject-338819/fyp_instance