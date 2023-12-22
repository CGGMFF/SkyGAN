# Dockerfile for running StyleGAN3/SkyGAN

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 pandas

WORKDIR /workspace

RUN (printf '#!/bin/bash\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
ENTRYPOINT ["/entry.sh"]

RUN pip install imgui glfw pyopengl

RUN conda install opencv=4.5.0 -c conda-forge
RUN git clone https://github.com/pybind/pybind11 /pybind11
RUN pip install diskcache ipdb

# copy and build the clear sky model
COPY sky_image_generator.cpp /workspace/
COPY sky_image_generator.h /workspace/
COPY ArPragueSkyModelGroundXYZ /workspace/ArPragueSkyModelGroundXYZ
RUN c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(cd /pybind11 && python3 -m pybind11 --includes) -I/opt/conda/include/opencv4/ -I/opt/conda/lib/python3.8/site-packages/numpy/core/include/ -L/opt/conda/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs

RUN apt-get update && apt-get install -y --no-install-recommends x11-apps xauth mesa-utils # ... for glxgears
