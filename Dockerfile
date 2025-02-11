FROM python:3.8

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set up the environment
ENV PATH /opt/conda/bin:$PATH

WORKDIR /main

COPY environment.yaml .
RUN conda env create -f environment.yaml
RUN echo "source activate control" > ~/.bashrc
ENV PATH /opt/conda/envs/control/bin:$PATH

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]