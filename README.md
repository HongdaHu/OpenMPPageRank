# OpenMPPageRank
A parallel PageRank program in OpenMP.

***Requirements***:
Write a parallel PageRank program in OpenMP (please see Resources Part to find the requirement of input Web graph file). Assume that only one of the threads reads the file. The pagerank values are initialized to a normalized identity vector by all the threads, and then updated using a matrix vector product. The process continues until the page ranks do not change significantly (you should explicitly give the condition or threshold in your report).

***Run***:
1.ENVIRONMENT:
Ubuntu 13.10
2.DATASET:
Using task1/src/facebook_combined.txt
3.OUTPUT:
Output_Task1.txt
4.COMMAND:
cd <local path>/task1/src
export OMP_NUM_THREADS=4
mpic++ task1.c -fopenmp
./a.out

***Implementation***:
1.Read the Facebook graph into a hash map(a->b and b->a), which is equal to a adjacency list;
2.Initialize the adjacency matrix(N*N), walk through the adjacency list and set the matrix, make sure each column's values' sum is 1;
3.Initialize the rank vector R(N), start value is 1/N;
4.Parallel compute the inner product of matrix's every row with R(N), renew the R(N);
5.For each renew of the R(N), get the largest change of the element of R(N), when the largest change of R(N) is less than the threshold, it will stop and done.
Performance:
When the rank value changes less than 0.000001e-8, the iterator will stop.
And it will run 80 times.
