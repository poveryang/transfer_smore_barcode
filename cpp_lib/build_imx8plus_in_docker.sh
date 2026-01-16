#!/bin/bash

# Get the directory of target project
script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
echo "Project directory: $project_dir"

# Clear the build and dist directories
rm -rf "$project_dir/build/imx8plus"
rm -rf "$project_dir/dist"

# Build the docker image
container_id=$(docker run -it \
    --platform linux/amd64 \
    -d \
    -v "$project_dir":/workspace \
    compiler:imx8plus)

# Build the project in the docker container
docker exec "$container_id" bash -c "
    /usr/bin/cmake \
        -DROI_PROJECTOR_BUILD_TEST=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/workspace/dist \
        -DCMAKE_CXX_COMPILER='/opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++' \
        -S /workspace/cpp_lib \
        -B /workspace/build/imx8plus &&
    /usr/bin/cmake \
        --build /workspace/build/imx8plus \
        --target install \
        -- -j 16
"

# Stop and remove the docker container
docker stop "$container_id"
docker rm "$container_id"