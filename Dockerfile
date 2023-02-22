from osgeo/gdal:ubuntu-small-latest

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
RUN pip install h3 --pre

RUN python3 -m pytest
RUN mypy yirgacheffe
RUN python -m pylint yirgacheffe