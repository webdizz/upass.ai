FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 as base

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING UTF-8
ENV CONDA_ENV upass
ENV CONDA_PATH /opt/conda/bin:/opt/conda/envs/$CONDA_ENV/bin/
ENV PATH=$CONDA_PATH:$PATH
COPY ./environment.yml /environment.yml

FROM base as conda
RUN apt-get update \
	&& apt-get install -y --no-install-recommends wget \ 
				bzip2 \
				ca-certificates \
				libglib2.0-0 \
				libxext6 \
				libsm6 \
				libxrender1 \
	 			libgomp1 \
	 			vim \
	 			sudo \
				gcc \
	&& wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
	&& /bin/bash ~/miniconda.sh -b -p /opt/conda \
	&& rm ~/miniconda.sh \
	&& echo '. /opt/conda/etc/profile.d/conda.sh' > /etc/profile.d/conda.sh \
	&& . /etc/profile.d/conda.sh \
	&& conda update -n base conda \
	&& conda clean --all -y \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/cache/apt

FROM conda as condaenv
RUN . /etc/profile.d/conda.sh \
	&& conda env create -f /environment.yml  \
	#&& conda install --name $CONDA_ENV cuda100 -c pytorch \
	&& conda clean --all -y \
	&& rm -rf ~/.cache/pip \
	&& conda activate $CONDA_ENV

# Some notes on how to do it later https://github.com/python-pillow/Pillow/blob/master/depends/ubuntu_14.04.sh
#FROM condaenv as fastoptim
#RUN conda uninstall --name $CONDA_ENV -y --force pillow pil jpeg libtiff \
#	&& pip uninstall -y pillow pil jpeg libtiff \
#	&& conda install --name $CONDA_ENV -c conda-forge libjpeg-turbo \
#	&& CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd \
#   	&& conda install --name $CONDA_ENV -y jpeg libtiff \
#	&& conda clean --all -y

FROM condaenv as userspace
RUN groupadd appuser \
	&& useradd --create-home -r --shell=/bin/bash -g appuser appuser \
	&& usermod -aG sudo appuser \
	&& echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
	&& mkdir -p /opt/notebooks \
	&& mkdir /opt/data \
	&& chown appuser:appuser /opt/notebooks \
	&& chown -R appuser:appuser /opt/conda \
	&& chown -R appuser:appuser /opt/data \
	&& echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

VOLUME "/opt/notebooks"
VOLUME "/opt/data"

USER appuser
CMD ["jupyter", "notebook", "--notebook-dir=/opt/notebooks", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]

EXPOSE 8888
