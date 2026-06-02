FROM ghcr.io/osgeo/gdal:ubuntu-small-3.12.4

RUN apt-get update -qqy && \
	apt-get install -qy \
		git \
		python3-pip \
		python3-venv \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/cache/apt/*

COPY ./ /root/
WORKDIR /root/

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install gdal[numpy]==3.12.4
RUN pip install mlx
RUN pip install -e .[dev,matplotlib]

RUN python3 -m pytest -vv
RUN mypy yirgacheffe
RUN python3 -m pylint yirgacheffe
