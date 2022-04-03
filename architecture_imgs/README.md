# Architecture comparison

# Approach 1. Proxy to self-managed environment

![image description](./Architecture%201.png)
In this approach, a proxy will act as a gateway for interacting with the environments. If the user has no environments, it will make use of the back-end's API to create one and will be connected to it. On the other hand, If an environment is running, the proxy will directly connect the user to the self-managed environment.

When the environment is created, it comes with it's own back-end and controller. This environment becomes a stand-alone server with which the user interacts with. The server has specific APIs to update the environment EC2 instances, run simulations, and update data. The server provides all the functionality, but runs independently from the main back-end.

The Controller makes sure instances are healthy and is responsible to restart instances in case of them going down. It also takes care of logging and maintaining the session running or closing it if unused.

## Advantages

- Each user gets their own back-end server to interact with.
- Each environment has it own controller focusing solely on the instances of that environment and user.
- Mitigates traffic between main-server and users. Main-server is solely used to manage the CRUD operations for the self-managed environment.
- Easier to maintain due to local scope.

## Disadvantages

- Single point of failure in the controller of the environment - if controller goes down, the whole environment is compromised and destroyed.
- Communication issues with main-server in case of failure - might not be able to inform about error.
- Could use databases directly, but would quickly mean database gets targeted by multiple servers, potentially increasing load over time.
- Spinning up a server every time a new environment is created is costly, both in terms of performance, and instances used on AWS. Moreover, a user might have multiple environments running, which would imply multiple server would be created for each such environment.

# Approach 2. Main controllers with individual controllers

![image description](./Architecture%202.png)
In this approach, the proxy is used again as a gateway between users and their environments. However, instead of creating a self-managed environment, there are a suite of 'main' controllers that handle these environments.

In this approach, users will interact with the environment through the main controllers, which in turn make changes to environments by communicating with individual controllers. This approach resembles a Master-Slave approach, where the main controllers act as masters, and individual environment's controllers as slaves.

Whenever a command is issued, a main controller will be selected to forward the request to the specific environment's controller. The controller will receive the message and handle the request just like in the first approach.

However, the main difference is that only main controllers write to databases, and they provide a failure safety by constantly checking the health of controllers, making sure responses are processed and by providing multiple handlers for incoming requests, mitigating the global single point of failure from architecture 1.

## Advantages

- Users have multiple points of entry to their environments through the main controllers.
- Single point of failure of connection to environments is mitigated - state is known by main controllers and, in case of local controller failure, it can re-create it.
- Reduced database communication - only main controllers communicate with database, and they can do so in batches.

## Disadvantages

- Even more costly to integrate - more servers are needed for main controllers
- More complex in terms of integration and implementation.
- Single point of failure at the level of environments.
- Would need a load balancer or reverse proxy to redirect traffic accordingly.

# Approach 3. Main controllers without individual controllers.

![image description](./Architecture%203.png)
Similar to approach 3, this approach would make use of a suite of main controllers handling environments for users.

However, environments would be solely controlled by the current main controller interacting with them, without having a local controller.

When a request is issued by a user, a main controller is selected to handle the request. The main controller gets information about the environment from the database or its local cache (i.e. getting IPs of machines) and sends out request to each machine or coordinates the learning process.

## Advantages

- Only a few machines are required for the main controllers, thus the cost and number of instances is greatly reduced.
- Creating an environment is a matter of having a main controller start the environment, without the complexity and cost of creating a local controller responsible for each environment's instance.
- In case of main controller failure, any other controller can pick up the request and handle it - might imply greater complexity by having a mechanism for checking if request was handled.
- No single point of failure in environments.
- Faster creation times.

## Disadvantages

- Highly complex and difficult to get right
- Error/Failure handling comes with complex mechanisms.
- Increased database usage and size for storing information about each environment.
- Introduction of load balancer to distribute work to main controllers.

## Other approaches and important details

- The architectures described did not include all the relevant details for databases. This was done to simplify the approach of comparison and analysis.
- The architecture did not discuss or included in diagrams scalability or design of front-end, back-end and proxy. This are separate things we will consider in a different part.
- Another potential approach would be to use the back-end server(s) to handle the connections to environments. Although this would greatly reduce the number of instances, it would also come with more load for the back-end server, more complexity on the back-end, and would increase the responsibility of the back-end server(s). This approach would be more inclined towards the monolithic approach to building software, which is not the approach we are aiming for. We consider that having specialized services (microservices) come with the complexity of proper integration and communication between services, but provide more flexibility, performance and scalability.
- The proxy in diagrams could be replaced by an equal load balancer that forwards requests based on the logic described in the approaches above. We are still open to how this connector will be implemented, but have considered either approaches are correct in during analysis.
