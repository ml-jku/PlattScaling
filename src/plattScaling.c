/* Copyright (C) 2018 Andreas Mayr, Guenter Klambauer*/
/*Licensed under GNU General Public License v3.0 (see https://github.com/ml-jku/PlattScaling/blob/master/LICENSE)*/


/*
gcc -c -Wall -Werror -fpic plattScaling.c -std=c99
gcc -shared -o plattScaling.so plattScaling.o



import ctypes as ct
plattLib = ct.cdll.LoadLibrary('/system/user/mayr/scratch/platt/plattScaling.so')

import numpy as np
plattLib = np.ctypeslib.load_library('plattScaling','/system/user/mayr/scratch/platt')

plattLib.plattScaling.restype=None
plattLib.plattScaling.argtypes=[
np.ctypeslib.ctypes.c_int32,
np.ctypeslib.ndpointer(dtype=np.float64),
np.ctypeslib.ndpointer(dtype=np.int32),
np.ctypeslib.ctypes.c_double,
np.ctypeslib.ctypes.c_double,
np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_double),
np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_double)
]


target=np.concatenate([np.repeat(False, 10), np.repeat(True, 20)]).astype(np.int32)
#out=target+np.random.normal(0.0, 0.3, len(target))
out=target+np.sin(np.arange(0, len(target)))*0.3+0.3
A=np.ctypeslib.ctypes.c_double(0.0)
B=np.ctypeslib.ctypes.c_double(0.0)
Ap=np.ctypeslib.ctypes.pointer(A)
Bp=np.ctypeslib.ctypes.pointer(B)
plattLib.plattScaling(len(target), out, target, float(np.sum(np.logical_not(target))), float(np.sum(target)), Ap, Bp)

A.value
-3.712755256429975

B.value
1.6075770717508315




target=c(rep(FALSE, 10), rep(TRUE, 20))
#out=target+rnorm(length(target), 0, 0.3)
out=target+sin(seq(0, length(target)-1))*0.3+0.3
dyn.load("plattScaling.so")
.Call("plattScaling", out, target, as.numeric(sum(!target)), as.numeric(sum(target)))
dyn.unload("plattScaling.so")
 */

#include "plattScaling.h"
#include <limits.h>
#include <stdlib.h>
#include <math.h>

void plattScaling(int len, double *deci, int *label, double prior0, double prior1, double* Aret, double* Bret, double* sucRet) {
	int maxiter=100; //Maximum number of iterations
	double minstep=1e-10; //Minimum step taken in line search
	double sigma=1e-12; //Set to any value > 0
	//Construct initial values: target support in array t,
	// initial function value in fval
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	len=prior1+prior0; // Total number of data
	double* t=(double*)malloc(sizeof(double)*len);
	for(int i=0; i<len; i++) {
		if (label[i] > 0)
			t[i]=hiTarget;
		else
			t[i]=loTarget;
	}
	double A=0.0;
	double B=log((prior0+1.0)/(prior1+1.0));
	double suc=0.0;
	double fval=0.0;
	for(int i=0; i<len; i++) {
		double fApB=deci[i]*A+B;
		if (fApB >= 0)
			fval += t[i]*fApB+log(1+exp(-fApB));
		else
			fval += (t[i]-1)*fApB+log(1+exp(fApB));
	}
	for(int it=1; it<maxiter; it++) {
		//Update Gradient and Hessian (use H’ = H + sigma I)
		double g1=0.0;
		double g2=0.0;
		double h11=sigma;
		double h22=sigma;
		double h21=0.0;
		for(int i=0; i<len; i++) {
			double fApB=deci[i]*A+B;
			double p;
			double q;
			if (fApB >= 0) {
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else {
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			double d2=p*q;
			h11 += deci[i]*deci[i]*d2;
			h22 += d2;
			h21 += deci[i]*d2;
			double d1=t[i]-p;
			g1 += deci[i]*d1;
			g2 += d1;
		}
		if (abs(g1)<1e-5 && abs(g2)<1e-5) { //Stopping criteria
				suc=1.0;
				break;
		}
		//Compute modified Newton directions
		double det=h11*h22-h21*h21;
		double dA=-(h22*g1-h21*g2)/det;
		double dB=-(-h21*g1+h11*g2)/det;
		double gd=g1*dA+g2*dB;
		double stepsize=1;
		while (stepsize >= minstep){ //Line search
			double newA=A+stepsize*dA;
			double newB=B+stepsize*dB;
			double newf=0.0;
			for(int i=0; i<len; i++) {
				double fApB=deci[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB+log(1+exp(-fApB));
				else
					newf += (t[i]-1)*fApB+log(1+exp(fApB));
			}
			if (newf<fval+0.0001*stepsize*gd){
				A=newA;
				B=newB;
				fval=newf;
				break;
			}
			else
				stepsize /= 2.0;
		}
		if (stepsize < minstep){
			suc=0.0;
			break;
		}

		if (A > 0) {
			A=0.0;
		}
	}
	free(t);
  *Aret=A;
  *Bret=B;
  *sucRet=suc;
}
