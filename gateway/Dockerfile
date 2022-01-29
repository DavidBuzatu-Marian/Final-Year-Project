FROM node:16.13.2-alpine

# RUN
# Use latest version of npm for package management
RUN npm i npm@latest -g

# RUN
# Create the working directory in the container
# Used by all the other services in the compose file
RUN mkdir -p /usr/src/gateway

# WORKDIR
# Set the working directory to the global usage
WORKDIR /usr/src/gateway

# COPY
# Copy the package json, lock and everything related
# Crucial to install the dependencies
COPY package*.json ./

# RUN
# Install the dependencies silently (to not fill the logs)
RUN npm install --silent

# COPY
# Copy all the files to the working directory
# Action needed in order to have all the static files
# And serve them
COPY . .

# EXPOSE
# Expose the default port 5002
EXPOSE 5002

# USER
# Change to node user in order to remove privileges
USER node

CMD [ "npm", "run", "dev" ]