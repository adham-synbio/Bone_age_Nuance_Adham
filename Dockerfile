FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

MAINTAINER Adham Alkhadrawi <adham.alkhadrawi@mgh.harvard.edu>

RUN rm /etc/apt/sources.list.d/cuda.list

# Set environment variables (PIN Recommendation)
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install git, python3, and pip (Recommended Regardless of base image)
RUN apt -y update && apt -y install git python3 python3-pip

# Ensure venv and dependencies installed (PIN Requirement)
RUN apt -y install libjpeg-dev python3-distutils python3-venv python3-gdcm wget zlib1g-dev && \
    apt-get autoclean && \
    apt-get clean

# Set up virtual environment (PIN Requirement)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Update setuptools (PIN Requirement)
RUN python3 -m pip install --upgrade --no-cache-dir --ignore-installed setuptools

# Set version of pip to prevent pip install issues
RUN python3 -m pip install --upgrade pip

# Install ai_service (PIN Requirement)
COPY ai_service-2.0.3.37-py3-none-any.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/ai_service-2.0.3.37-py3-none-any.whl  ${EXTRA_PYTHON_PACKAGES} && \
    rm /tmp/ai_service-2.0.3.37-py3-none-any.whl

# Install library requirements (After installing AIM Service to prevent bugs)
ARG EXTRA_PYTHON_PACKAGES
COPY requirements.txt /tmp/
RUN python3 -m pip install --upgrade --no-cache-dir ${EXTRA_PYTHON_PACKAGES} -r /tmp/requirements.txt

# Set UTC Time
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Clean up apt information of data efficiency
RUN rm -rf /var/lib/apt/lists/*

ENV DEBUG=YES
ENV KEEP_FILES=YES

# Ensure all messages reach the console
ENV PYTHONUNBUFFERED=1

# Activate Virtual Environment for AI Service
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Move application runtime codebase to app directory
WORKDIR /app

COPY ./pin_deploy_boneage_ai_service.py /app/
COPY ./preprocess.py /app/
COPY ./version.py /app/
COPY ./model.py /app/
COPY ./seg_model.pt /app/resources/
COPY ./M_model.pt /app/resources/
COPY ./F_model.pt /app/resources/

CMD ["python3", "pin_deploy_boneage_ai_service.py"]

