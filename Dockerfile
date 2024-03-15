from ghcr.io/osgeo/gdal:ubuntu-small-latest

RUN apt-get update -qqy && \
	apt-get install -qy \
		git \
		python3-pip \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /var/cache/apt/*

COPY ./ /root/
WORKDIR /root/

RUN pip install numpy
RUN pip install pylint mypy pytest types-setuptools
RUN pip install h3==4.0.0b2

RUN python3 -m pytest -vv
# RUN mypy yirgacheffe
RUN python -m pylint yirgacheffe
