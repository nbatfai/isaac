/**
 * @brief ISAAC - deep Q learning with neural networks for predicting the next sentence of a conversation
 *
 * @file qlc.cu
 * @author  Norbert Bátfai <nbatfai@gmail.com>
 * @version 0.0.1
 *
 * @section LICENSE
 *
 * Copyright (C) 2015 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 * SAMU
 *
 * The main purpose of this project is to allow the evaluation and
 * verification of the results of the paper entitled "A disembodied
 * developmental robotic agent called Samu Bátfai". It is our hope
 * that Samu will be the ancestor of developmental robotics chatter
 * bots that will be able to chat in natural language like humans do.
 *
 */

#include "qlc.h"

__device__ double
sigmoid ( double x )
{
    return 1.0/ ( 1.0 + exp ( -x ) );
}

__device__ double
prcp ( int j, int nu, double *newu, double *u, double *w )
{
    newu[j] = 0.0;
    for ( int k = 0; k < nu; ++k ) {
        newu[j] += w[j*nu+k] * u[k];
    }
    return sigmoid ( newu[j] );
}

__global__ void
layer_kernel ( int nu, double *newu, double *u, double *w )
{
    int j = blockIdx.x;
    newu[j] = prcp ( j, nu, newu, u, w );
}

void cuda_layer ( int i, int* n_units,   double **units,   double ***weights )
{
    double *device_newu;
    cudaMalloc ( ( void ** ) &device_newu, n_units[i] * sizeof ( double ) );

    double *device_u;
    cudaMalloc ( ( void ** ) &device_u, n_units[i-1] * sizeof ( double ) );
    cudaMemcpy ( device_u, units[i-1],
                 n_units[i-1]*sizeof ( double ), cudaMemcpyHostToDevice );

    double *device_w;
    cudaMalloc ( ( void ** ) &device_w, n_units[i] * n_units[i-1] * sizeof ( double ) );
    for ( int wi = 0; wi<n_units[i]; ++wi ) {

        cudaMemcpy ( device_w+wi*n_units[i-1], weights[i-1][wi],
                     n_units[i-1]*sizeof ( double ), cudaMemcpyHostToDevice );
    }

    dim3 grid ( n_units[i] , 1 );
    layer_kernel <<< grid, 1 >>> ( n_units[i-1], device_newu, device_u, device_w );

    cudaMemcpy ( units[i], device_newu,
                 n_units[i]*sizeof ( double ), cudaMemcpyDeviceToHost );

    cudaFree ( device_newu );
    cudaFree ( device_u );
    cudaFree ( device_w );

}
