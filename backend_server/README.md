# Main Controller

## Description

As explained in the Architecture section of the report. the back-end is reponsible to handle the interaction with the database and controllers. The back-end server will serve clients with information related to the environments, such as status, available data and data distribution, and training logs. Moreover, the back-end will ensure interactions with environments are realised through controllers.

The back-end server is currently a single-point-of-failure in our architecture, however we could easily adopt the methodology of controllers and use multiple back-end servers behind a load balancer to handle requests. We decided not to do so in our current implementation due to the time constraints and added complexity. However, we find it important to be mentioned as our flexible design allows for such upgrades.

Overall, the back-end has been developed using JavaScript and uses [NodeJS](https://nodejs.org/en/) and [ExpressJS](https://expressjs.com) to handle incoming requests. Moreover, it uses [Mongoose](https://mongoosejs.com) to handle interactions with MongoDB, [BullJS](https://github.com/OptimalBits/bullm) to handle task scheduling and it also contain snippets of HashiCorp Configuration Language for [Terraform](https://www.terraform.io) and Dockerfiles to allow building [Docker](https://www.docker.com) images and containers.

The back-end contains the necessary configuration file for creating Main Controller instance on Google Cloud, however we do not automatically use them in our approach. They could be added to a possible health checker to be able to re-create crashed controllers.

## Structure

The code has been split up into folders that are independent and have single responsiblity within their effects.

- config: contains a default json file with specific configuration values required to run the Redis database or make requests to the load balancer.
- hooks: contains a set of functions that are used to perform operations relating to file uploads, environment interaction and url parsing. The functions in 'environment.js' relate to how job responses are handled and how their body is created. The 'upload.js' defines a function to delete local files. Finally, 'url.js' defines a function that is used to parse a url by removing its query parameters.
- logs: contains a file defining the functions required by our logger **Morgan** to be able to log incoming requests and errors. The logs folder also contains a 'logs' folder to store the logs.
- models: defines the set of database schemas for environment addresses, data distribution, training distribution and training logs. these models are used to define the right tables and entries in MongoDB.
- node_modules: ignored by git. It is created by the 'npm' package manager when installing all the dependencies defined in 'package.json'.
- routes: contains the files responsible to define and handle the back-end endpoints. More details in the _endpoints_ section.
- temp: the folder is currently empty and should remain empty. It is used during file upload to locally save the files being sent over.
- terraform: contains the necessary terraform configuration files to create Main Controllers on Google Cloud.
- workers: contains sets of functions responsible to handle different types of tasks. Environment workers define handlers for the creation, deletion and addition of data to environments. The health workers process tasks related to health checks on controllers. Finally, the model workers define the task processors for model creation, losses and training. We have also added a [strategy.js](/backend_server/workers/strategy.js) file to incorporate the Strategy Design pattern in some of our routes. Due to some of our endpoints having similar functionalities - e.g., create a task and serve its status - we found that we could generalise this behaviour using the strategy pattern. Thus, endpoints are pattern matched using regex by using the available options from our strategy map. The strategy map is an object that maps endpoints to their corresponding task queue.
- Files outside of folders: some files were left outside of folders due to required organisation and import style of ExpressJS. These files include Dockerfiles (which need to be in the parent root of the folder to build the images using the right context), the `package.json` file containing requirements used by NodeJS and the `server.js` file that defines the server code.

## Endpoints

We provide a list of endpoints that can be targeted in the back-end below:

### Dataset endpoints

- **/api/dataset/distribution** (GET): Finds all the environments' data distribution for a given user id. This endpoint is blocking and not using tasks.
- **/api/dataset/training/distribution** (GET): Find all the environments' distribution of data for training. This value denotes how the training data should be distributed to each instance. This endpoint is blocking and not using tasks.

### Environment endpoints

- **/api/environment/create** (POST): The endpoint is responsible to create the task for environment creation and submit it to the queue.
- **/api/environment/create:id** (GET): Returns the status for the given environment create task id.

- **/api/environment/delete** (POST): The endpoint is responsible to create the task for environment deletion and submit it to the queue.
- **/api/environment/delete:id** (GET): Returns the status for the given environment delete task id.

- **/api/environment/dataset/data** (POST): Submits the task for adding the environment's training data to the dataset queue.
- **/api/environment/dataset/validation** (POST): Submits the task for adding the environment's validation data to the dataset queue.
- **/api/environment/dataset/test** (POST): Submits the task for adding the environment's testing data to the dataset queue.
- **/api/environment/dataset/distribution** (POST): Submits the task for adding the environment's training distribution to the dataset queue.
- **/api/environment/** (GET): Returns all the environments pertaining to a user id. This endpoint is blocking and not using tasks.
- **/api/trainlogs/** (GET): Returns all the envionemnts' train logs for given user id. This endpoint is blocking and not using tasks.
- **/api/environment/dataset/:id** (GET): Returns the status for the given dataset task id. It can be any task related to datasets.

### Health endpoints

- **/api/health/status** (POST): The endpoint is responsible to create the task for environment health check and submit it to the queue.
- **/api/health/status:id** (GET): Returns the status for the given environment health check task id.

### Model endpoints

Uses the aforementioned [strategy.js](/backend_server/workers/strategy.js) maps to dynamically match the endpoint. Although the code responds to any incoming request of the form `/api/model/`, it has the following 4 handlers:

- **/api/model/create**
- **/api/model/train**
- **/api/model/loss**
- **/api/model/status**
  All these handlers are _POST_ requests and they simply create a task and add it to the corresponding queue. To get the queue, the code uses the strategy map which stores the specific Queue for the endpoint - e.g., '/api/model/create' -> modelCreateQueue. There are also corresponding _GET_ requests for these endpoints to get the task status given the id:
- **/api/model/create:id**
- **/api/model/train:id**
- **/api/model/loss:id**
- **/api/model/status:id**

- **/api/model/download** (GET) - This endpoint can be used to download the trained model after the Federated Learning training rounds finish.

## Docker

The Dockerfile used to build the back-end image performs some required initial set-up to be able to work with terraform, and install the necessary node libraries.

We have also extensively used a [docker-compose](./docker-compose.yml). In the case of the back-end server, it was mandatory for the docker-compose file to be used as the back-end also uses a `Redis` database, which has to be created before it is started.

To run the controller locally, execute:

```Docker
docker-compose up --build
```

By default, the back-end should be found on localhost:5005, while the redis container should be found on localhost:6379.

## Tests

Unfortunately, due to the complexity of mocking that the back-end requires, we have not implemented tests for this component.
