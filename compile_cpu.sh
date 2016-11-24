# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#echo $TF_INC
MINICONDA="/vol/atlas/homes/gt108/miniconda/envs/menpo/"
TF_INC="$MINICONDA/lib/python3.5/site-packages/tensorflow/include"

rm -f trace_norm.so

g++ -std=c++11 -shared trace_norm.cc -o trace_norm.so -fPIC -I $TF_INC -I $MINICONDA/include -L $MINICONDA/lib
