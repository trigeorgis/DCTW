# Deep Canonical Time Warping

A Tensorflow implementation of the Deep Canonical Time Warping.

    Deep Canonical Time Warping
    G. Trigeorgis, M. A. Nicolaou, S. Zafeiriou, B. Schuller.
    Proceedings of IEEE International Conference on Computer Vision & Pattern Recognition (CVPR'16).
    June 2016.

# Installation Instructions

We are an avid supporter of the Menpo project (http://www.menpo.org/) which we use
in various ways throughout the implementation.

In general, as explained in [Menpo's installation instructions](http://www.menpo.org/installation/),
it is highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.

Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n dctw python=3.5
$ source activate dctw
```

**Step 2:** Install [TensorFlow](https://www.tensorflow.org/) following the
official [installation instructions](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html).
For example, for 64-bit Linux, the installation of CPU-only, Python 3.5 TensorFlow involves:
```console
(dctw)$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
```


**Step 3:** Install [menpo](https://github.com/menpo/menpo) from the _menpo_ channel as:
```console
(dctw)$ conda install -c menpo menpo
```

**Step 4:** Compile the extra `TraceNormOp` op.
```console
(dctw)$ bash compile_cpu.sh
```

When you are done you can go through the example in [demo.ipynb](https://github.com/trigeorgis/tf_dctw/blob/master/demo.ipynb).
