# ------------------------------------------------------------------------------
# Author: Bailey Allison
# ST558 Project 3 - Dockerfile
#
# This Docker image runs a Plumber API for predicting diabetes status using a 
# tuned random forest model fit to the 2015 BRFSS Diabetes Health Indicators data.
#
# The API contains:
#   • /pred      – returns a predicted class and probabilities 
#   • /info      – returns developer info and project URL
#   • /confusion – returns a confusion matrix plot
#
# The container starts plumber on port 8000 by default.
# ------------------------------------------------------------------------------

# Use a rocker image that already has R and tidyverse
FROM rocker/tidyverse:4.3.2

# Install required R packages
RUN R -e "install.packages(c('plumber', 'tidymodels', 'ranger'))"

# Set working directory inside the container
WORKDIR /app

# Copy plumber file and data into the image
COPY plumber.R /app/
COPY diabetes_binary_health_indicators_BRFSS2015.csv /app/

# Expose the port that plumber will run on
EXPOSE 8000

# Command to start the Plumber API when the container runs
CMD ["R", "-e", "pr <- plumber::plumb('plumber.R'); pr$run(host = '0.0.0.0', port = 8000)"]
