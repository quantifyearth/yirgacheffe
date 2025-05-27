from ghcr.io/osgeo/gdal:ubuntu-small-3.11.0

RUN apt-get update -qqy && \
	apt-get install -qy \
		git \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/cache/apt/*

COPY ./ /root/
WORKDIR /root/

RUN pip config set global.break-system-packages true
RUN pip install gdal[numpy] scikit-image torch
RUN pip install pylint mypy pytest types-setuptools
RUN pip install h3

RUN python3 -m pytest -vv
# RUN mypy yirgacheffe
RUN python -m pylint yirgacheffe
