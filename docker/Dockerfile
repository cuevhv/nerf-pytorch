# copying docker with pytorch 1.12
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Make sure we use cuda
ENV export FORCE_CUDA="1"

# adding to user to local group
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# CMD nvidia-smi
# WORKDIR /app
# install useful packages
RUN apt-get update
RUN apt-get install -y git make wget

# RUN apt-get update
# RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
#         apt-get clean

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# RUN bash ~/miniconda.sh -b -p $HOME/miniconda
# RUN $HOME/miniconda/bin/conda init
# RUN source ~/.bashrc
# ENV PATH /root/miniconda/bin:$PATH

# RUN apt-get install -y vim-tiny


# install anaconda
RUN apt-get update
# RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
#         apt-get clean
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


RUN conda update --all

# RUN conda create -y -n face_anim python=3.9
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate face_anim" >> ~/.bashrc
ENV PATH /opt/conda/envs/face_anim/bin:$PATH
ENV CONDA_DEFAULT_ENV face_anim


CMD ["nginx", "-g", "daemon off;"]
