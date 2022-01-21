#! /bin/bash
apt update
apt -y install docker.io
docker pull eu.gcr.io/finalyearproject-338819/fyp_main_controller
docker run -p 5000:5000 eu.gcr.io/finalyearproject-338819/fyp_main_controller