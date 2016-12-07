TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
MINICONDA=$(dirname $(dirname $(dirname $(dirname $(dirname $TF_INC)))))

rm -f trace_norm.so

g++ -std=c++11 -shared trace_norm.cc -o trace_norm.so -fPIC -I $TF_INC -I $MINICONDA/include -L $MINICONDA/lib
