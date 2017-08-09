#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include "mpi.h"

using namespace std;

void decomposition(int size, int rank, int m, int &my_number_of_subintervals, int &my_first_midpoint) {
	int p = m / size;
	int r = m % size;
	if (rank < r)
	{
		my_number_of_subintervals = p + 1;
		my_first_midpoint = rank * (p + 1);
	}
	else
	{
		my_number_of_subintervals = p;
		my_first_midpoint = r * (p + 1) + (rank - r) * p;
	}
}

double evaluate_f_of_x(double x, int k , double* alpha) {

	double f = 0;

	for (int i = k; i >= 0; i--) {
		f += alpha[i] * pow(x, i);
	}

	return f;
}


double integrate(double a, double b, int m, int k, double* alpha)
{
	double x;
	double f_of_x;
	double h = (b - a) / m;
	double integral = 0.0;

	x = a;
	f_of_x = evaluate_f_of_x(x, k, alpha);
	integral += 0.5 * f_of_x;

	x = a + m * h;
	f_of_x = evaluate_f_of_x(x, k, alpha);
	integral += 0.5 * f_of_x;

	for (int i = 1; i < m; i++)
	{	
		x = a + i * h;
		f_of_x = evaluate_f_of_x(x, k, alpha);
		integral += f_of_x;
	}

	integral *= h;
	return integral;
}



int main(int argc, char *argv[]){

	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int k;
	double *alpha;
	double a, b;
	int m;
	
	int pack_size;
	char *pack_buff;
	int pack_position;

	int my_number_of_subintervals;
	int my_first_midpoint;


	if (rank == 0) {

		ifstream file;

		if (argc > 1) {
			file.open(argv[1]);
		}
		else file.open("test5b.in");

		if (!file.good()) {
			cout << "Unable to open a file: " << argv[1] << endl;
			MPI_Finalize();
			return 0;
		}


		file.seekg(12);
		file >> k;

		alpha = new double[k + 1];

		file.seekg(15, file.cur);

		for (int i = 0; i <= k; i++) {
			file >> alpha[i];
		}


		file.seekg(15, file.cur);

		file >> a;
		file >> b;


		file.seekg(15, file.cur);

		file >> m;

		file.close();

		int tmp_pack_size;

		MPI_Pack_size(2, MPI_INT, MPI_COMM_WORLD, &tmp_pack_size);
		pack_size = tmp_pack_size;

		MPI_Pack_size(k + 1 + 2, MPI_DOUBLE, MPI_COMM_WORLD, &tmp_pack_size);
		pack_size += tmp_pack_size;

		pack_buff = new char [pack_size];

		pack_position = 0;
		MPI_Pack(&k, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(alpha, k + 1, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&a, 1, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&b, 1, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
		MPI_Pack(&m, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);

		MPI_Bcast(&pack_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(pack_buff, pack_size, MPI_PACKED, 0, MPI_COMM_WORLD);

	}
	else{
		
		MPI_Bcast(&pack_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

		pack_buff = new char[pack_size];

		MPI_Bcast(pack_buff, pack_size, MPI_PACKED, 0, MPI_COMM_WORLD);

		pack_position = 0;

		MPI_Unpack(pack_buff, pack_size, &pack_position, &k, 1, MPI_INT, MPI_COMM_WORLD);

		alpha = new double[k + 1];

		MPI_Unpack(pack_buff, pack_size, &pack_position, alpha, k+1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(pack_buff, pack_size, &pack_position, &a, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(pack_buff, pack_size, &pack_position, &b, 1, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Unpack(pack_buff, pack_size, &pack_position, &m, 1, MPI_INT, MPI_COMM_WORLD);
	}
		

	double h = (b - a) / m;


	decomposition(size, rank, m, my_number_of_subintervals, my_first_midpoint);

	double integration = integrate(a + my_first_midpoint*h, a + (my_first_midpoint + my_number_of_subintervals)*h, my_number_of_subintervals, k, alpha);

	double result = 0;

	MPI_Reduce(&integration, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0){

		cout.precision(20);
		
		cout << "\nResult: " << result << endl;

		double Ia = 0;
		double Ib = 0;

		for (int i = k; i >= 0; i--) {
			Ia += (alpha[i] * pow(a, i+1)) / (i + 1);
			Ib += (alpha[i] * pow(b, i+1)) / (i + 1);
		}

		Ib -= Ia;

		cout << "\nAnalytical result: " << Ib << endl;

	}

	MPI_Finalize();
	return 0;
}
