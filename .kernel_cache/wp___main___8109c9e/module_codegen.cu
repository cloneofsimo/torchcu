
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)



extern "C" __global__ void simple_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_a,
    wp::array_t<wp::vec_t<3,wp::float32>> var_b,
    wp::array_t<wp::float32> var_c)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float32>* var_1;
        wp::vec_t<3,wp::float32> var_2;
        wp::vec_t<3,wp::float32> var_3;
        wp::vec_t<3,wp::float32>* var_4;
        wp::vec_t<3,wp::float32> var_5;
        wp::vec_t<3,wp::float32> var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 1.0;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // forward
        // def simple_kernel(a: wp.array(dtype=wp.vec3),                                          <L 5>
        // tid = wp.tid()                                                                         <L 10>
        var_0 = builtin_tid1d();
        // x = a[tid]                                                                             <L 13>
        var_1 = wp::address(var_a, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // y = b[tid]                                                                             <L 14>
        var_4 = wp::address(var_b, var_0);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // r = wp.dot(x, y)                                                                       <L 17>
        var_7 = wp::dot(var_3, var_6);
        // r = 1.0 / (1.0 + wp.exp(-r))                                                           <L 19>
        var_9 = wp::neg(var_7);
        var_10 = wp::exp(var_9);
        var_11 = wp::add(var_8, var_10);
        var_12 = wp::div(var_8, var_11);
        // c[tid] = r                                                                             <L 22>
        wp::array_store(var_c, var_0, var_12);
    }
}

extern "C" __global__ void simple_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float32>> var_a,
    wp::array_t<wp::vec_t<3,wp::float32>> var_b,
    wp::array_t<wp::float32> var_c,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_a,
    wp::array_t<wp::vec_t<3,wp::float32>> adj_b,
    wp::array_t<wp::float32> adj_c)
{
    for (size_t task_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         task_index < dim.size;
         task_index += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        wp::vec_t<3,wp::float32>* var_1;
        wp::vec_t<3,wp::float32> var_2;
        wp::vec_t<3,wp::float32> var_3;
        wp::vec_t<3,wp::float32>* var_4;
        wp::vec_t<3,wp::float32> var_5;
        wp::vec_t<3,wp::float32> var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 1.0;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3,wp::float32> adj_1 = {};
        wp::vec_t<3,wp::float32> adj_2 = {};
        wp::vec_t<3,wp::float32> adj_3 = {};
        wp::vec_t<3,wp::float32> adj_4 = {};
        wp::vec_t<3,wp::float32> adj_5 = {};
        wp::vec_t<3,wp::float32> adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        //---------
        // forward
        // def simple_kernel(a: wp.array(dtype=wp.vec3),                                          <L 5>
        // tid = wp.tid()                                                                         <L 10>
        var_0 = builtin_tid1d();
        // x = a[tid]                                                                             <L 13>
        var_1 = wp::address(var_a, var_0);
        var_2 = wp::load(var_1);
        var_3 = wp::copy(var_2);
        // y = b[tid]                                                                             <L 14>
        var_4 = wp::address(var_b, var_0);
        var_5 = wp::load(var_4);
        var_6 = wp::copy(var_5);
        // r = wp.dot(x, y)                                                                       <L 17>
        var_7 = wp::dot(var_3, var_6);
        // r = 1.0 / (1.0 + wp.exp(-r))                                                           <L 19>
        var_9 = wp::neg(var_7);
        var_10 = wp::exp(var_9);
        var_11 = wp::add(var_8, var_10);
        var_12 = wp::div(var_8, var_11);
        // c[tid] = r                                                                             <L 22>
        // wp::array_store(var_c, var_0, var_12);
        //---------
        // reverse
        wp::adj_array_store(var_c, var_0, var_12, adj_c, adj_0, adj_12);
        // adj: c[tid] = r                                                                        <L 22>
        wp::adj_div(var_8, var_11, var_12, adj_8, adj_11, adj_12);
        wp::adj_add(var_8, var_10, adj_8, adj_10, adj_11);
        wp::adj_exp(var_9, var_10, adj_9, adj_10);
        wp::adj_neg(var_7, adj_7, adj_9);
        // adj: r = 1.0 / (1.0 + wp.exp(-r))                                                      <L 19>
        wp::adj_dot(var_3, var_6, adj_3, adj_6, adj_7);
        // adj: r = wp.dot(x, y)                                                                  <L 17>
        wp::adj_copy(var_5, adj_4, adj_6);
        wp::adj_load(var_4, adj_4, adj_5);
        wp::adj_address(var_b, var_0, adj_b, adj_0, adj_4);
        // adj: y = b[tid]                                                                        <L 14>
        wp::adj_copy(var_2, adj_1, adj_3);
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_a, var_0, adj_a, adj_0, adj_1);
        // adj: x = a[tid]                                                                        <L 13>
        // adj: tid = wp.tid()                                                                    <L 10>
        // adj: def simple_kernel(a: wp.array(dtype=wp.vec3),                                     <L 5>
        continue;
    }
}

