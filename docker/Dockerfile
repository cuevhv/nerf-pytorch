FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Make sure we use cuda
ENV export FORCE_CUDA="1"

# adding to user to local group
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

# remove this in case of conflicts when running apt-get update
USER user 

# CMD nvidia-smi
# WORKDIR /app
# install useful packages
RUN apt-get update
RUN apt-get install -y git make wget

# install anaconda
RUN apt-get update
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# create conda env based on yml config file
RUN conda update --all
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate face_anim" >> ~/.bashrc
ENV PATH /opt/conda/envs/face_anim/bin:$PATH
ENV CONDA_DEFAULT_ENV face_anim


CMD ["nginx", "-g", "daemon off;"]
