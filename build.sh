if [ "$1" = "build" ]; then
    mkdir -p _build
    cd _build
    cmake ..
    make
fi

if [ "$1" = "run" ]; then
    mkdir -p _build
    cd _build
    cmake ..
    make main
    ./main
fi

if [ "$1" = "clean" ]; then
    rm -rf _build
fi

if [ "$1" = "tests" ] || [ $# = 0 ]; then
    mkdir -p _build
    cd _build
    cmake ..
    make tests
    ./tests
fi

if [ "$1" = "help" ]; then
  echo "./build { build | run | clean | tests}"
fi
