DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. && make -j11

cd ${DIR}