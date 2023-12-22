# Dockerfile for testing/experimenting with the Prague sky model

FROM cschranz/gpu-jupyter:v1.4_cuda-11.0_ubuntu-20.04_python-only
USER root
RUN chmod -R a+rw /home/jovyan # hack: allow anyone to read/write ... allow running the installed tools as any user
USER $NB_UID

RUN pip install easydict opencv-python zstd
RUN conda install opencv

RUN git clone https://github.com/pybind/pybind11
COPY sky_image_generator.cpp /home/jovyan
COPY sky_image_generator.h /home/jovyan
COPY ArPragueSkyModelGroundXYZ /home/jovyan/ArPragueSkyModelGroundXYZ
RUN c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp $(cd ~/pybind11 && python3 -m pybind11 --includes) -I/opt/conda/include/opencv4/ -I/opt/conda/lib/python3.8/site-packages/numpy/core/include/ -L/opt/conda/lib/ sky_image_generator.cpp -o sky_image_generator$(python3-config --extension-suffix) -lopencv_core -lopencv_imgcodecs

RUN pip install torchinfo ipdb diskcache pytest graphviz
