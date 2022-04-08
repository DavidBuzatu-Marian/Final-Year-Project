# Gateway

## Description

Our gateway was added to our architecture as a means of providing centralised authentication. We aimed at keeping the responsibility of each of our components solely for the things they have to do, thus having authentication checked at each of the components - e.g., instance, back-end - would have implied repeating code and adding more complexity and responsiblity.

Overall, the gateway has been developed using JavaScript and uses [NodeJS](https://nodejs.org/en/) and [ExpressJS](https://expressjs.com) to handle incoming requests. Its main responsibility is to authenticate users using [PassportJS](https://www.passportjs.org), and on successful authentication proxy users to the back-end server.

To handle authentication, passport prepares a cookie token which we store in a session. Passport has been set up to save user data in MongoDB using the defined [user model](/gateway/models/User.js).

Moreover, we make use of the library [http-proxy-middleware](https://www.npmjs.com/package/http-proxy-middleware) to proxy our requests to the back-end. For this, we have defined [routes](/gateway/routes/routes.js), which is an array of possible route handlers in the proxy. We had to do this in order to set some headers used by the other components, while also handling file uploads.

Finally, the gateway also used numerous libraries to handle requests - e.g., [form-data](https://www.npmjs.com/package/form-data) used to pass multipart data (files), [morgan](https://www.npmjs.com/package/morgan) to handle logs, [multer](https://www.npmjs.com/package/multer) to handle file uploads, etc. A list of required packages can be found in [package.json](/gateway/package.json).

## Structure

The code has been split up into folders that are independent and have single responsiblity within their effects.

- config: contains a config JSON file used to get various environment values - e.g., which MongoDB connection to use. Moreover, it defines a function for managing the connection to MongoDB.
- hooks: contains useful functions to help with functionality used in route handling. In our case, it defines a function to delete local files.
- logs: contains a file defining the functions required by our logger **Morgan** to be able to log incoming requests and errors. The logs folder also contains a 'logs' folder to store the logs.
- middleware: contains an [authentication middleware](/gateway/middleware/auth.js) used to check for authentication on sensitive endpoints, i.e. endpoints that require authentication.
- models: defines the User model used in MongoDB to store user data.
- node_modules: a folder required by nodejs. This folder contains all the required packages.\
- routes: defines a set of route handlers for endpoints related to 'auth' and 'environment'. The other files are used by the proxy middleware to handle redirects.
- temp: the folder is currently empty and should remain empty. It is used during file upload to locally save the files being sent over.
- files outside of folders: they define the Docker configuration files, packages to be installed and server file.

## Endpoints

We provide a list of endpoints that can be targeted inside instances below:

### Proxy endpoints

- All requests that are not related to environment data or user authentication (see below) will be proxied to the back-end server. All endpoints are first checked for authentication, and if they pass, they will be redirected to the back-end. If a user is not authenticated, an HTTP status error is returned to the user.

### Environment endpoints

- **/dataset/data** (POST) - we specifically handle the dataset upload endpoint without the proxy middleware due to implementation constrains. The middleware is not capable of proxying files with the request, thus we had to handle this particular case ourselves.

### Authentication endpoints

- **/auth/login** (POST) - used to login a user. It returns the user id and email.
- **/auth/authenticated** (GET) - used to check if a user is currently authenticated. It returns the user id and email
- **/auth/register** (POST) - the endpoint is used to register a user given their email address and password. This endpoint does not automatically login users!
- **/auth/logout** (POST) - the endpoint is responsible for logging out the user by deleting the user session and invalidating the authentication token.

## Docker

The Dockerfile used to build the instance images performs some required initial set-up to install the necessary npm libraries.

We have also extensively used a [docker-compose](./docker-compose.yml) file during development to spin up the container for the gateway. The compose files could also be further used when deploying the gateway to any host.

To run the gateway locally, execute:

```Docker
docker-compose up --build .
```

By default, the gateway should run on localhost:5002. Please check your Docker Desktop for more details.

## Tests

Unfortunately, due to the lack of time and inexperience with frontend testing, we have not implemented tests for this component.
