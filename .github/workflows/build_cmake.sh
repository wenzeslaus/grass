#!/usr/bin/env bash

# The make step requires something like:
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PREFIX/lib"
# further steps additionally require:
# export PATH="$PATH:$PREFIX/bin"

# fail on non-zero return code from a subprocess
set -e

# print commands
set -x

if [ -z "$1" ]
then
    echo "Usage: $0 PREFIX"
    exit 1
fi

# non-existent variables as an errors
set -u

export INSTALL_PREFIX=$1

mkdir build
cd build

cmake \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DPostgreSQL_TYPE_INCLUDE_DIR="/usr/include/postgresql" \
    ..

make
make install
