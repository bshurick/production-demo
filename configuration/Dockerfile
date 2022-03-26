FROM ubuntu:21.10

# Set a docker label to advertise multi-model support on the container
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    libgomp1 \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3.9 \
    python3-dev \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN python get-pip.py

# Install dependencies via production package
# TODO: Install demo package in environment and launch entrypoint
COPY . pkg
RUN cd pkg && pip3 --no-cache-dir install .
RUN mkdir /opt/ml && mkdir /opt/ml/model && mv pkg/data/trained_model /opt/ml/model/trained_model

ENTRYPOINT ["/usr/local/bin/entrypoint"]
