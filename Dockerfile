FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING UTF-8
ENV CONDA_ENV upass
ENV PATH /opt/conda/bin:/opt/conda/envs/$CONDA_ENV/bin/:$PATH

RUN apt-get update \
	&& apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
	&& apt-get install -y libgomp1 vim \
	&& echo 'export PATH=$PATH' > /etc/profile.d/conda.sh \
	&& wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
	&& /bin/bash ~/miniconda.sh -b -p /opt/conda \
	&& rm ~/miniconda.sh \
	&& . /etc/profile.d/conda.sh \
	&& conda update -n base conda

COPY ./environment.yml /environment.yml

RUN conda env create -f /environment.yml  \
	&& conda install --name $CONDA_ENV cuda92 -c pytorch \
	&& conda clean --all -y \
	&& groupadd appuser \
	&& useradd --create-home -r --shell=/bin/bash -g appuser appuser \
	&& mkdir -p /opt/notebooks \
	&& chown appuser:appuser /opt/notebooks \
	&& chown -R appuser:appuser /opt/conda/envs \
	&& mkdir /opt/data \
	&& chown -R appuser:appuser /opt/data

VOLUME "/opt/notebooks"

USER appuser
CMD ["jupyter", "notebook", "--notebook-dir=/opt/notebooks", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]

EXPOSE 8888
