#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include "mpi.h"

using namespace std;

void openfile(int c, char *argv[], double **&matrix, int &row, int &col) {

	ifstream file;

	file.open(argv[c]);

	if (!file.is_open()){
		cout << "Unable to open a file: " << argv[c] << endl;
		MPI_Finalize();
		return;
	}

	file >> row;
	file >> col;

	matrix = new double*[row];

	for (int i = 0; i < row; ++i)
		matrix[i] = new double[col];


	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			file >> matrix[i][j];

	file.close();
}

void sendtofile(int c, char *argv[], double **matrix, int row, int col){
	ofstream file;

	file.open(argv[c]);

	file << row << " " << col << endl;

	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			file << matrix[i][j] << " ";
		}
		file << endl;
	}
	file.close();
}

void decomposition(int p, int r, int q, int size, int &nh, int &nv, int &a, int &b){
	bool nice, supernice=false;
	int bufor = (p + r)*q;
	int lnh, lnv, la, lb;
	for (lnh = 1; lnh <= size; lnh++){
		nice = true;
		if (p % lnh == 0){
			la = p / lnh;
			if (size % lnh == 0){
				lnv = size / lnh;
				if (r % lnv == 0){
					lb = r / lnv;
				}else nice = false;
			}else nice = false;
		}else nice = false;

		if (nice){
			if ((la + lb)*q <= bufor){
				a = la;
				b = lb;
				nh = lnh;
				nv = lnv;
				bufor = (la + lb)*q;
				supernice = true;
			}
		}
	}
	if (!supernice){
		cout << "Decomposition failed, wrong number of processes" << endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
		return;
	}
}

void calculationij(int rank, int nv, int a, int b, int **&values){
	int imin = a*(rank / nv);
	int imax = imin + a - 1;
	int jmin = b*(rank % nv);
	int jmax = jmin + b - 1;
	cout << "\nRank: " << rank << endl;
	cout << "i= <" << imin << "," << imax << ">" << endl;
	cout << "j= <" << jmin << "," << jmax << ">" << endl;
	values[0][0] = imin;
	values[0][1] = imax;
	values[1][0] = jmin;
	values[1][1] = jmax;
}

void matrixtosend(int rank, int **values, double **A, double **B, int a, int b, int q, double **&matrixa, double **&matrixb, int &pack_size, char *&pack_buff, int &pack_position){



	for (int i = 0; i < a; i++){
		for (int j = 0; j < q; j++){
			matrixa[i][j] = A[i+values[0][0]][j];
		}
	}


	for (int i = 0; i < q; i++){
		for (int j = 0; j < b; j++){
			matrixa[i][j] = B[i][j+values[1][0]];
		}
	}

	int tmp_pack_size;

	MPI_Pack_size(3, MPI_INT, MPI_COMM_WORLD, &tmp_pack_size);
	pack_size = tmp_pack_size;

	MPI_Pack_size((a+b)*q, MPI_DOUBLE, MPI_COMM_WORLD, &tmp_pack_size);
	pack_size += tmp_pack_size;

	pack_buff = new char[pack_size];

	pack_position = 0;
	MPI_Pack(&a, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
	MPI_Pack(&b, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
	MPI_Pack(&q, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);

	for (int i = 0; i < a; i++){
		MPI_Pack(matrixa[i], q, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
	}

	for (int i = 0; i < q; i++){
		MPI_Pack(matrixb[i], b, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
	}

	MPI_Send(&pack_size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
	MPI_Send(pack_buff, pack_size, MPI_PACKED, rank, 0, MPI_COMM_WORLD);

}

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int a, b, q;
	int p, q2, r;


	double** matrixa;
	double** matrixb;
	double** matrixc;

	MPI_Status status;

	int pack_size;
	char *pack_buff;
	int pack_position;

	int ***Values;

	Values = new int**[size];
	for (int i = 0; i < size; i++){
		Values[i] = new int*[2];
		for (int j = 0; j < 2; j++)
			Values[i][j] = new int[2];
	}

	if (rank == 0) {

		double **A, **B;

		int nh, nv;


		if (argc != 4){
			cout << "Error, wrong number of parameters!" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 0;
		}

		openfile(1, argv, A, p, q);
		openfile(2, argv, B, q2, r);

		if (q != q2) {
			cout << "Error, wrong matrix!" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
			return 0;
		}

		decomposition(p, r, q, size, nh, nv, a, b);

		cout << "\nn= " << size << endl << "p= " << p << endl << "r= " << r << endl << "nh= " << nh << endl << "nv= " << nv << endl << "a= " << a << endl << "b= " << b << endl;

		for (int i = 0; i < size; i++){
			calculationij(i, nv, a, b, Values[i]);
		}

		cout << "\nTotal data to send size : " << (a + b)*q*sizeof(double) << " B" << endl;

		matrixa = new double *[a];
		for (int i = 0; i < a; i++)
			matrixa[i] = new double[q];

		matrixb = new double *[q];
		for (int i = 0; i < q; i++)
			matrixb[i] = new double[b];


		for (int i = 0; i < a; i++){
			for (int j = 0; j < q; j++){
				matrixa[i][j] = A[i + Values[0][0][0]][j];
			}
		}


		for (int i = 0; i < q; i++){
			for (int j = 0; j < b; j++){
				matrixa[i][j] = B[i][j + Values[0][1][0]];
			}
		}


		for (int s = 1; s < size; s++){
			for (int i = 0; i < a; i++){
				for (int j = 0; j < q; j++){
					matrixa[i][j] = A[i + Values[s][0][0]][j];
				}
			}


			for (int i = 0; i < q; i++){
				for (int j = 0; j < b; j++){
					matrixa[i][j] = B[i][j + Values[s][1][0]];
				}
			}

			int tmp_pack_size;

			MPI_Pack_size(3, MPI_INT, MPI_COMM_WORLD, &tmp_pack_size);
			pack_size = tmp_pack_size;

			MPI_Pack_size((a + b)*q, MPI_DOUBLE, MPI_COMM_WORLD, &tmp_pack_size);
			pack_size += tmp_pack_size;

			pack_buff = new char[pack_size];

			pack_position = 0;
			MPI_Pack(&a, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
			MPI_Pack(&b, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
			MPI_Pack(&q, 1, MPI_INT, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);

			for (int i = 0; i < a; i++){
				MPI_Pack(matrixa[i], q, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
			}

			for (int i = 0; i < q; i++){
				MPI_Pack(matrixb[i], b, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
			}

			MPI_Send(&pack_size, 1, MPI_INT, s, 0, MPI_COMM_WORLD);
			MPI_Send(pack_buff, pack_size, MPI_PACKED, s, 0, MPI_COMM_WORLD);
		}

	}
	else{

		MPI_Recv(&pack_size, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		pack_buff = new char[pack_size];

		MPI_Recv(pack_buff, pack_size, MPI_PACKED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		pack_position = 0;

		MPI_Unpack(pack_buff, pack_size, &pack_position, &a, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(pack_buff, pack_size, &pack_position, &b, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Unpack(pack_buff, pack_size, &pack_position, &q, 1, MPI_INT, MPI_COMM_WORLD);

		matrixa = new double *[a];
		for (int i = 0; i < a; i++)
			matrixa[i] = new double[q];

		matrixb = new double *[q];
		for (int i = 0; i < q; i++)
			matrixb[i] = new double[b];

		for (int i = 0; i < a; i++){
			MPI_Unpack(pack_buff, pack_size, &pack_position, matrixa[i], q, MPI_DOUBLE, MPI_COMM_WORLD);
		}

		for (int i = 0; i < q; i++){
			MPI_Unpack(pack_buff, pack_size, &pack_position, matrixb[i], b, MPI_DOUBLE, MPI_COMM_WORLD);
		}


	}

	matrixc = new double *[a];

	for (int i = 0; i < a; i++){
		matrixc[i] = new double[b];
	}

	for (int k = 0; k < q; k++){
		for (int i = 0; i < a; i++){
			for (int j = 0; j < b; j++){
				matrixc[i][j] += matrixa[i][k] * matrixb[k][j];
			}
		}
	}

	if (rank != 0){
		MPI_Pack_size(a*b, MPI_DOUBLE, MPI_COMM_WORLD, &pack_size);
		pack_buff = new char[pack_size];

		pack_position = 0;
		for (int i = 0; i < a; i++){
			MPI_Pack(matrixa[i], b, MPI_DOUBLE, pack_buff, pack_size, &pack_position, MPI_COMM_WORLD);
		}
		MPI_Send(&pack_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(pack_buff, pack_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD);

	}
	if (rank==0){
		
		double** C = new double*[p];

		for (int i = 0; i < p; ++i)
			C[i] = new double[r];


		for (int i = 0; i < a; i++){
			for (int j = 0; j < b; j++){
				C[i + Values[0][0][0]][j + Values[0][1][0]] = matrixc[i][j];
			}
		}
			MPI_Recv(&pack_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			pack_buff = new char[pack_size];

			MPI_Recv(pack_buff, pack_size, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			for (int i = 0; i < a; i++){
				MPI_Unpack(pack_buff, pack_size, &pack_position, matrixc[i], b, MPI_DOUBLE, MPI_COMM_WORLD);
			}

			for (int i = 0; i < a; i++){
				for (int j = 0; j < b; j++){
					C[i + Values[status.MPI_SOURCE][0][0]][j + Values[status.MPI_SOURCE][1][0]] = matrixc[i][j];
				}
			}


		sendtofile(3, argv, C, p, r);
		free(C);

	}

	free(Values);
	

	MPI_Finalize();
	return 0;
}