FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
WORKDIR /workspace
COPY requirements-rtx5090.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check
CMD ["tail", "-f", "/dev/null"]
