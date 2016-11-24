#undef EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/Eigen/SVD"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/kernels/linalg_ops_common.h"

REGISTER_OP("TraceNorm")
.Input("inputs: float32")
.Output("trace: float32")
.Output("u: float32")
.Output("v: float32");


namespace tensorflow {

  class TraceNormOp : public OpKernel {
    public:
      explicit TraceNormOp(OpKernelConstruction* context)
        : OpKernel(context) {
        }

      using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      void Compute(OpKernelContext* context) override {
        const Tensor& input_t = context->input(0);
        const Matrix& input_m = Eigen::Map<const Matrix>(input_t.flat<float>().data(),
            input_t.dim_size(0),
            input_t.dim_size(1));

        auto options = Eigen::ComputeThinU | Eigen::ComputeThinV;
        Eigen::BDCSVD<Eigen::Matrix<float,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> svd(input_m, options);

        Tensor* output = nullptr;
        TensorShape output_shape({1,});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        TensorShape u_shape({input_t.dim_size(0), input_t.dim_size(0)});
        Tensor* u_t = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, u_shape, &u_t));
        
        TensorShape v_shape({input_t.dim_size(1), input_t.dim_size(1)});
        Tensor* v_t = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, v_shape, &v_t));
        
        *(output->flat<float>().data()) = svd.singularValues().sum();

        auto u_m = Eigen::Map<Matrix>(u_t->flat<float>().data(), u_t->dim_size(0), u_t->dim_size(1));
        auto v_m = Eigen::Map<Matrix>(v_t->flat<float>().data(), v_t->dim_size(0), v_t->dim_size(1));
        u_m = svd.matrixU();
        v_m = svd.matrixV();
      }

    private:

    TF_DISALLOW_COPY_AND_ASSIGN(TraceNormOp);
  };

  REGISTER_KERNEL_BUILDER(Name("TraceNorm").Device(DEVICE_CPU), TraceNormOp);
}
