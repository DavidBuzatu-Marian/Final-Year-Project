# Thesis
Thesis can be read [here](Thesis.pdf). 

# Project instructions

This file provides general information about the contents of the repository, the required software to be able to run the project and how to run the project.

Each individual code folder contains its own **README.md** file. Please consult those files for complete descriptions of the contents and workflows of the specific servers.

## Repository contents

This repository is split into numerous folders. We list the folders below:

- .vscode - Ignored on git. Contains a settings file used by visual studio code.
- archtiecture_imgs - Contains various diagrams representing architecture variants for our solution. It also contains a [README.md](/architecture_imgs/README.md) file that entails the thorough analysis of different architectures.
- backend_server: Contains the code that defines the back-end server component written in **JavaScript**.
- class_diagrams: Contains the class diagram for the factory classes used in the 'nn\_' files.
- frontend: Contains the code for the frontend written in **JavaScript**.
- gateway: Contains the code for the gateway component. It has been written in **JavaScript**.
- instance: Defines the code required by the instances. It has been developed using **Python**.
- json_examples: Contains a suite of JSON examples that can be used to create the \U-Net\* model and train it.
- load_balancer: Contains the code for the **nginx** load balancer.
- main_controller: Defines the code required by the controllers. It has been developed using **Python**.
- useful_images: Contains a set of diagrams, figures and flows which are used in our report. They can be inspected for further analysis.
- user_flows: Contains various user flows throughout the system which were used when designing our architecture.
- server_flows: Contains various system flows.

## Software requirements

To be able to run the servers in each aforementioned folder, it is essential to have [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine. Due to all of our servers being containerised, you do not need to install any other external dependencies or even interpreters for programming languages. Docker will install everything needed to run the servers in a container using an isolated environment which automatically installs dependencies. The exception to this is our front-end, which we were not able to containerise due to internal issues in NextJS. Thus, you will have to install [npm](https://www.npmjs.com) beforehand.

To run our end-to-end tests, you have to install [Postman](https://postman.com). To import our collection of endpoint tests, please follow the steps outlined in the official [tutorial](https://kb.datamotion.com/?ht_kb=postman-instructions-for-exporting-and-importing).

## How to run the project?

Once docker is installed, follow the instructions in each server's **README.md** file to see how to run the code. We are doing this due to some servers being handled with the `docker-compose` command, while others using `docker run`.

We do have to mention that it may not be able to run the whole functionality of the project due to restrictions imposed by Google Cloud and our personal account. We have set up [gcloud](https://cloud.google.com/sdk/gcloud) on our machines and used our personal account to use the cloud services. We cannot provide our authentication details here due to having direct access to our bank account and violating the privacy terms of Google Cloud. Should you want to set up a Google Cloud account, you can set up our project to run with your account instead. Our [terraform file](/main_controller/terraform/finalyearproject-338819-12b837ed8475.json) contains private authentication tokens that may be usable, but we have no means of testing they work. Please do not share such keys as they are directly connected to our account.

In case running the whole codebase is mandatory, please reach out to one of us for enabling Google Cloud interactions, as they cost real money. The codebase may function without the access to Google Cloud, but it cannot perform any task related to environments - such as creation.
