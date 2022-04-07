# Frontend

## Description

Our frontend was created to serve users using a GUI. We adhered to industry standards and built the frontend using JavaScript and the [React](https://react.com) framework. However, to prepare our frontend for future deployments, improve overall performance, add caching and more, we have used a layer of abstraction on top of React called [NextJS](https://nextjs.org). Nextjs is a framework that accelerates development with React, and that has especially helped us during development to have real-time updates without refresh for our frontend content.

The frontend helped us simplify the interaction users by having a graphical interface. We have created specialised pages for authentication, environment creation, dashboard - i.e., environment management -, and datasets. Then, we have modularised our code such that components could be re-used along different pages, within different settings. One such example is the [closableAlert](/frontend/components/alert/closableAlert.js) component, which we use on numerous pages, within multiple components, in order to display successful or error messages.

We have also put emphasis on how content is loaded and presented, so that users see progress spinners whenever longer tasks are executed - e.g., logging in, fetching environments, creating environments, etc. - and we gracefully handle most of errors with error messages.

Due to our time contraints and focus on back-end (logic) implementation, we have made use of [mui](https://mui.com) to design our interface. Mui provided out-of-the-box components and styling for various common things, such as forms, buttons, spinners, etc. However, we have also added our own flavor of styling to some components which can be seen by identifying _style=_ attributes in the code.

Finally, our implementation made use of numerous libraries to handle specific functionalities. These include, but are not limited to:

- axios: used to make asynchronous HTTP requests
- downloadjs: used to download files in an interactive window
- react-json-editor-ajrm: used to edit JSON while also checking for validity
- swr: used to manage the state for data around the front-end - e.g., environment information

## Structure

The code has been split up into numerous folders independent of each other. They are as follow:

- .next: a file ignored by git but used by nextjs.
- components: defines all the suite of component used in our pages. The folder contains numerous sub-folders that are individualised for specific types of components, including 'alert', 'auth', 'dashboard', etc.
- config: contains configuration files. We have only defined _local_ configuration, but our solution could use 'google-cloud' config while deployed on Google Cloud Compute Engine instances for example.
- hooks: defines a suite of functions that are used to fetch data related to datasets, environments and users.
- node_modules: a folder required by nodejs. This folder contains all the required packages.
- pages: defines the pages in our implementation as described in section 'Description'. There are a few pages which are required by nextjs - i.e., \_app, \_document - that we have added some configurations to, but that are required by nextjs to function properly.
- public: contains the generated static files for the frontend
- src: contains a theme file as defined by **mui**.
- styles: contains generic css and scss files for styling.
- files outside folders: general configuration files or server files kept outside due to their required scope.

## Docker

We have kept our frontend outside a Docker container due to some issues related to the version of Linux issued by the image. We had some issues with SWC and continued to run the frontend locally.

To run the frontend, execute:
`npm run dev -- -p 3001`

If the **node_modules** is not found, please run first:
`npm install`.

## Tests

Unfortunately, due to the lack of time and inexperience with frontend testing, we have not implemented tests for this component.
