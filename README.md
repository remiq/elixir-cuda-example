# Elixir CUDA Example

Example of controlling CUDA-using-CNode with Elixir.

## Usage

Compile Elixir lib

    $ make

Now try sub-chapters

### Basic CNode

Try basic CNode first - this blocks terminal

    $ make basic_cnode

Start iex

    $ make start_elixir

You can now send commands to CNode ie.

    iex> ElixirCudaExample.foo 1
    2

### Stand-alone CUDA program

Try to execute simple CUDA program

    $ make cuda_standalone
    mkdir -p bin
    /usr/local/cuda/bin/nvcc -g -G -Xcompiler -Wall cnode/cuda_standalone.cu -o bin/cuda_standalone
    ./bin/cuda_standalone
    0 + 0 = 0
    -1 + 1 = 0
    -2 + 4 = 2
    -3 + 9 = 6
    -4 + 16 = 12
    -5 + 25 = 20
    -6 + 36 = 30
    -7 + 49 = 42
    -8 + 64 = 56
    -9 + 81 = 72

### CUDA-using-CNode

Now both programs combined

    $ make cuda_cnode

Start iex

    $ make start_elixir

Send command to CNode

    iex> ElixirCudaExample.foo 1
    -1

    // while on CNode terminal
    Connected to e1@NEWBORN
    -1 + 0 = -1
    -1 + 1 = 0
    -1 + 2 = 1
    -1 + 3 = 2
    -1 + 4 = 3
    -1 + 5 = 4
    -1 + 6 = 5
    -1 + 7 = 6
    -1 + 8 = 7
    -1 + 9 = 8


## Installation

### CUDA SDK - Ubuntu 14.04

    $ sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    $ sudo apt-get update
    $ sudo apt-get install cuda

## Learn about CUDA

https://github.com/romain-jacotin/cuda
