#!/bin/bash

# set up script
if [ $# -ne 1 ]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

source ${THIS_DIR}/../build-setup.sh

CXX="nvcc"

# adjust to nvcc style link flags
LINK_FLAGS="${LINK_FLAGS/-Wl,-rpath,/-Xlinker -rpath -Xlinker }"

# build
${CXX} -std=c++14 -o ${THIS_DIR}/hpcg-benchmark \
    ${THIS_DIR}/hpcg-benchmark.cpp \
    ${THIS_DIR}/prolongation_kernel.cu \
    ${THIS_DIR}/restriction_kernel.cu \
    ${THIS_DIR}/matrix_generation_kernel.cu \
    -I${THIS_DIR}/../../include -I${BUILD_DIR}/include \
    -L${THIS_DIR} ${LINK_FLAGS}
