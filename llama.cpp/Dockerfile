ARG ONEAPI_VERSION=2025.0.0-0-devel-ubuntu22.04
ARG UBUNTU_VERSION=22.04

FROM intel/oneapi-basekit:$ONEAPI_VERSION AS build

ARG GGML_SYCL_F16=OFF
RUN apt-get update && \
    apt-get install -y git libcurl4-openssl-dev

WORKDIR /app

RUN apt-get install -y wget unzip 
RUN wget https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip -O /master.zip 
RUN unzip /master.zip -d /
RUN cp -rf /llama.cpp-master/* /app/

RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set" && \
        export OPT_SYCL_F16="-DGGML_SYCL_F16=ON"; \
    fi && \
    echo "Building with dynamic libs" && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_CURL=ON ${OPT_SYCL_F16} && \
    cmake --build build --config Release --target llama-server


FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/build/bin/llama-server /llama-server

ENV LC_ALL=C.utf8
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]