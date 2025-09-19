FROM ghcr.io/osgeo/gdal:ubuntu-small-3.11.0

RUN apt-get update -qqy && \
	apt-get install -qy \
		git \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/cache/apt/*

COPY ./ /root/
WORKDIR /root/

RUN pip config set global.break-system-packages true
RUN pip install gdal[numpy]==3.11.0
RUN pip install mlx
RUN pip install -e .[dev]

RUN python3 -m pytest -vv
RUN mypy yirgacheffe
RUN python -m pylint yirgacheffe
