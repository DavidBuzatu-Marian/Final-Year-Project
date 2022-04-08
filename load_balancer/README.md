# Load balancer

## Description

The load balancer we used is a configured [nginx](https://www.nginx.com) load balancer for our needs. To work with nginx in a containerised manner, we have used the default Docker image for [nginx](https://hub.docker.com/_/nginx) and configured it for our needs.

For that, we have defined the [default.conf](/load_balancer/default.conf) file. We had to define an 'upstream', which contains the IP addresses to balance the load to, i.e., the main controller addresses running locally. Because we have used at most 2 controllers locally due to our resource limits, it only contains two addresses.

Next, the 'server' block defines on which port to list to. The default 80 port is used for ... . We have also defined the 'client_max_body_size' to be 100MB to be able to send big datasets to the controllers.

The 'location' block defines different proxy configurations used when the requests are passed to the IP addresses mentioned in the 'upstream' block. We have defined various timeouts and headers passing configuration.

## Docker

We have not used a `docker-compose` file for nginx. Thus, to run the container locally, first build the image using:
`docker image build -t <image_name> .`
Followed by:
`docker run -p 3000:80 <image_name>`

## Tests

There are no tests for the load balancer.
