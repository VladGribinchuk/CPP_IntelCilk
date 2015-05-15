//// Example of numerical solution of the Dirichlet problem
//// for the Poisson equation using Seidel iterative method
////
//// As a domain of the jobs D some function U will be used
//// by the unit square
//// D = (x in [0..1], y in [0..1])
////
//// Boundary conditions:
//// U(0,y)=0 U(1,y)=y^2
//// U(x,0)=0 U(x,1)=x^2
////
//// Follow code implement parallel algorithm (using TBB library)

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"

// Lenght side of a square D will be A
const int A = 1;
double** initMatrix(int ny, int nx)
{
	// matrix will be consist of ny+1 rows and nx+1 cols
	double** matr = new double*[ny + 1];
	for (int i = 0; i <= ny; ++i)
	matr[i] = new double[nx + 1];
	// h - a step along the x and y axis
	double hx = A / (double)nx;
	double hy = A / (double)ny;
	double x = 0;
	double y = 0;
	for (int i = 0; i <= ny; ++i){
		y = i*hy;
		for (int j = 0; j <= nx; ++j){
			x = j*hx;
			matr[i][j] = 0.;
			if (i == ny)
			matr[i][j] = x*x;
			if (j == nx)
			matr[i][j] = y*y;
		}
	}
	return matr;
}
void cleanupMatrix(double** matrix, int ny)
{
// Remove and clear all the allocated memory
	for (int i = 0; i <= ny; ++i){
		delete[]matrix[i];
		matrix[i] = NULL;
	}
	delete matrix;
	matrix = NULL;
}
void outputMatrix(double** matrix, int ny, int nx)
{
	for (int i = 0; i <= ny; ++i){
		for (int j = 0; j <= nx; ++j)
			printf("%.3f ", matrix[i][j]);
			printf("\n");
	}
}

class Derichlet 
{
	const int N;
	double **const tmatrix;
	double **temp_matrix;

public:
	Derichlet(double** _matrix, int _N): tmatrix(_matrix), N(_N) 
	{
		temp_matrix = initMatrix(_N, _N);
	}
	void operator()(const tbb::blocked_range<int>& r) const
	{
		int begin = r.begin(),
			end = r.end();

		double dmax;
		
		//
		// Define precision(epsilon value)
		double e = .001;
		
		do
		{
			dmax = .0;
			for(int i = begin; i != end; ++i){
				for (int j = 1; j < N; ++j){
					temp_matrix[i][j] = 0.25*(tmatrix[i][j - 1] + tmatrix[i][j + 1] + tmatrix[i - 1][j] + tmatrix[i + 1][j]);
					double diff = fabs(temp_matrix[i][j] - tmatrix[i][j]);
					if (dmax < diff)
						dmax = diff;
					
					tmatrix[i][j] = temp_matrix[i][j];
				}
			}
		} while (dmax > e);
	}
};

int main()
{
	tbb::task_scheduler_init init;

	//
	// Graininess of x and y axis
	int N = 10;
	//
	// Graininess for parallel computing
	int grainsize = 10;
	//
	// Initialization of computing matrix
	double** matrix = initMatrix(N, N);
	
	tbb::parallel_for(tbb::blocked_range<int>(1,N,grainsize), Derichlet(matrix,N));

	//outputMatrix(matrix, N, N);

	cleanupMatrix(matrix, N);
	getchar();
	return EXIT_SUCCESS;
}
