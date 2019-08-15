ARG REGION=us-west-2
ARG DEVICE=gpu

FROM 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.0.0-${DEVICE}-py3

ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

COPY requirements.txt /opt/ml/code/

RUN pip install -r /opt/ml/code/requirements.txt