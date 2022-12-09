FROM python:3.8-slim
# Allow statements and log messages to immedately appear in the Knative logs
ENV PYTHONBUFFERED True

# Copy local code to the container image
RUN ECHO 'Copying local code to container image'
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ECHO 'Installing custom dependencies to make Flask API endpoint work'
# Install production dependencies
RUN pip install -r requirements.txt

RUN ECHO 'Executing gunicorn and binding to port set out by Google Cloud Run UI'
# Run the web service on container startup with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app