sudo systemctl start docker.service
sudo docker run -p 5000:5000 --shm-size=2gb eu.gcr.io/finalyearproject-338819/fyp_instance:latest