# FROM
# Install the apline version of nginx
FROM nginx:1.21.6-alpine

USER root


# RUN
# Remove the default configuration file
RUN rm /etc/nginx/conf.d/default.conf

# COPY
# Copy the virtual host file to the container
COPY ./default.conf /etc/nginx/conf.d/default.conf

# # RUN
# # Create sites-enabled folder
# RUN mkdir /etc/nginx/sites-enabled

# # RUN
# # Create symbolic link for load_balancer site
# RUN ln -s /etc/nginx/sites-available/controllers-host /etc/nginx/sites-enabled/controllers-host


# CMD
# Run the default command for nginx image
CMD ["nginx", "-g", "daemon off;"]