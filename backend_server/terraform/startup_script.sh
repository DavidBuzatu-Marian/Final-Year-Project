#! /bin/bash

sudo apt update
sudo apt -y install docker.io
sudo gcloud auth configure-docker -q
sudo docker pull eu.gcr.io/finalyearproject-338819/fyp_main_controller:latest
sudo docker run -p 5000:5000 eu.gcr.io/finalyearproject-338819/fyp_main_controller