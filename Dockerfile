FROM python:3.8-slim

# Setting enviroment
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONPATH=.

# Setup backage
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip 
# Create user
RUN groupadd nonroot \
    && useradd -g nonroot nonroot


# Copy and setup requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Change to noroot
USER nonroot

# Move all code to /app
COPY . /app
WORKDIR /app/

# Export port
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "-k", "uvicorn.workers.UvicornWorker", "--reload", "api:api"]