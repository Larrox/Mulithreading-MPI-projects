#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int main(int argc, char *argv[]){

	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double times[4];

	double start_time = MPI_Wtime();
	double time = start_time;
	double b_time;

	char fileName[] = "v04.dat";

	int vectors;
	int floats;
	int vectors_per_n;
	int startingvector;


	bool mpi = false;

	float *data;

	int i,j;

	if (!mpi){
		FILE *f;
		f = fopen(fileName, "r");
		if (f == NULL){
			printf("ERROR with opening file");
			MPI_Finalize();
			return 0;
		}
		int filesize;

		fseek(f, 0, SEEK_END); // seek to end of file
		filesize = ftell(f); // get current file pointer
		fseek(f, 0, SEEK_SET); // seek back to beginning of file

		vectors = filesize / 40;

		floats = vectors * 3;

		data = (float*)malloc(floats*(sizeof(float)));


		for (i = 0; i < floats; i++){
			fscanf(f, "%f", data + i);
		}
		fclose(f);

		b_time = MPI_Wtime();
		times[0] = b_time - time;
		time = b_time;


		if (vectors%size != 0){

			if (rank < (vectors%size)){
				vectors_per_n = (vectors / size) + 1;
			}
			else vectors_per_n = vectors / size;

		}
		else vectors_per_n = vectors / size;


		if (vectors%size != 0){
			if (rank < (vectors%size)){
				startingvector = rank*(vectors / size) + rank;
			}
			else startingvector = rank*(vectors / size) + (vectors % size);

		}
		else startingvector = rank*(vectors / size);

	}
	else{

		MPI_File file;
		MPI_Offset offset;

		if (MPI_File_open(MPI_COMM_SELF, fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS)
		{
			printf("File error\n");
			MPI_Abort(MPI_COMM_WORLD, 1);

			return -1;
		}

		MPI_File_get_size(file, &offset);

		vectors = offset / 40;
		floats = vectors * 3;


		if (vectors%size != 0){

			if (rank < (vectors%size)){
				vectors_per_n = (vectors / size) + 1;
			}
			else vectors_per_n = vectors / size;

		}
		else vectors_per_n = vectors / size;


		if (vectors%size != 0){
			if (rank < (vectors%size)){
				startingvector = rank*(vectors / size) + rank;
			}
			else startingvector = rank*(vectors / size) + (vectors % size);

		}
		else startingvector = rank*(vectors / size);

		data = (float*)malloc(vectors_per_n*3*(sizeof(float)));

		MPI_File_seek(file, startingvector*40, MPI_SEEK_SET);

		char *buff = (char*)malloc(vectors_per_n * 40 * sizeof(char));
		MPI_File_read(file, buff, vectors_per_n * 40, MPI_CHAR, MPI_STATUS_IGNORE);


		for (i = 0; i < vectors_per_n * 3; i++){
			data[i] = strtod(buff, &buff);
		}

		MPI_File_close(&file);

		b_time = MPI_Wtime();
		times[0] = b_time - time;
		time = b_time;
	}

	

	//printf("\nJestem mirek nr: %i, a moj poczatek to %i, a noja ilosc to %i\n", rank, startingvector, vectors_per_n);

	float avresults[4] = { 0.0, 0.0, 0.0, 0.0 };

	if (!mpi){

		for (i = startingvector * 3; i < (startingvector + vectors_per_n) * 3; i += 3){

			avresults[0] += sqrt(pow(data[i], 2) + pow(data[i + 1], 2) + pow(data[i + 2], 2));
			avresults[1] += data[i];
			avresults[2] += data[i + 1];
			avresults[3] += data[i + 2];

		}
	}
	else{

		for (i = 0; i < vectors_per_n * 3; i += 3){

			avresults[0] += sqrt(pow(data[i], 2) + pow(data[i + 1], 2) + pow(data[i + 2], 2));
			avresults[1] += data[i];
			avresults[2] += data[i + 1];
			avresults[3] += data[i + 2];
		}
	}

	float sum[4] = { 0.0, 0.0, 0.0, 0.0 };

	b_time = MPI_Wtime();
	times[1] = b_time - time;
	time = b_time;

	MPI_Reduce(avresults, sum, 4, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	b_time = MPI_Wtime();
	times[2] = b_time - time;
	times[3] = b_time - start_time;


	double *all_times = (double*)malloc(4 * size * sizeof(double));

	MPI_Gather(times, 4, MPI_DOUBLE, all_times, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0){

		printf("Length: %1.8f\nx: %1.8f y: %1.8f z: %1.8f\n", sum[0] / vectors, sum[1] / vectors, sum[2] / vectors, sum[3] / vectors);

		FILE *fp;
		char filename[sizeof "vresults.txt"];
		sprintf(filename, "vresults.txt");
		fp = fopen(filename, "w");


		int i = 0;
		for (i = 0; i < size; i++)
		{
			fprintf(fp, "timings (proc %i):\n", i);
			fprintf(fp, "readData: %1.8f\n", all_times[i*4]);
			fprintf(fp, "processData: %1.8f\n", all_times[i * 4 + 1]);
			fprintf(fp, "reduceResults: %1.8f\n", all_times[i * 4 + 2]);
			fprintf(fp, "total: %1.8f\n", all_times[i * 4 + 3]);
			fprintf(fp, "\n");
		}


		fclose(fp);
	}
	free(data);
	free(all_times);
	MPI_Finalize();
	return 0;
}
