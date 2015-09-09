//
//  A parallel PageRank program in OpenMP
//
//  Created by Hongda Hu on 4/9/14.
//  Copyright (c) 2015 Hongda. All rights reserved.
//

#include <iostream>
//#include <hash_map>
#include <fstream>
#include <omp.h>
#include <string>
#include <ext/hash_map>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#define CHUNKNUN 500
using namespace std;
using namespace __gnu_cxx;
	//key, value pair struct
	struct kvpairs{int id;float pagerank;};

	//Operator Overloading, Comparator, sort by id, Ascending
	bool operator < (const kvpairs &kva,const kvpairs &kvb){
			if (kva.id>kvb.id)
				return 0;
			if (kva.id<kvb.id)
				return 1;
	}

	//read file to the hashmap graph
	hash_map< int, vector<int> > readFileToHashMap (){
		int in,out;  
	
		hash_map< int, vector<int> > adjMap(4100);
	
		ifstream infile;   
		infile.open("facebook_combined.txt",ios::in);  
		while(!infile.eof()){            // until the end of file  
			infile>>in>>out;
			cout<<in<<" "<<out<<endl;

			//add edge: in_out
			if(adjMap.find(in)==adjMap.end()){
				vector<int> text;
				text.push_back(out);
				adjMap.insert(make_pair(in,text));
			}
			else{
				adjMap[in].push_back(out);
			}

			//add edge: out_in
			if(adjMap.find(out)==adjMap.end()){
				vector<int> text;
				text.push_back(in);
				adjMap.insert(make_pair(out,text));
			}
			else{
				adjMap[out].push_back(in);
			}
		}
		infile.close();
		return adjMap;
	} 
	
	//print the hashGraph out!
	void printHashMap(hash_map< int, vector<int> > adjMap){
		hash_map<int, vector<int> >::iterator  iter;
		vector<int>::iterator viter;
		vector<int> pvect;
		for(iter = adjMap.begin(); iter != adjMap.end(); iter++){
			cout<<"####"<<iter->first<<endl;

			pvect=iter->second;
			for ( viter = pvect.begin() ; viter != pvect.end() ; viter++ ){
				cout<<*viter<<" ";
			}		
			cout<<endl<<"Total number="<<pvect.size()<<endl;
		}
	}

	//initialize the M by traversaling adjMap!
	void initM(hash_map< int, vector<int> > adjMap, float ** M){

		hash_map< int, int> num2name(4100);
		hash_map<int, vector<int> >::iterator  iter;
		int num=0;
		int name;
		for(iter = adjMap.begin(); iter != adjMap.end(); iter++){
			name=iter->first;
			//add num2name
			num2name.insert(make_pair(num,name));
			num++;
		}

		vector<int>::iterator viter;
		vector<int> pvect;
		float ni=0.0;
		float vi=0.0;
		int i,j,tid;
		int pnum;

		#pragma omp parallel default(none) shared(adjMap,M,num2name) private(pnum,i,j,tid,ni,vi,viter,pvect)
		{
			#pragma omp for schedule(dynamic) nowait
			for(pnum = 0; pnum < num2name.size(); pnum++){
				tid = omp_get_thread_num();
				//printf("%d(%d) ",pnum,tid);
				j=num2name[pnum];
				pvect=adjMap[j];
				ni=pvect.size();
				if (!ni){
					continue;
				}
				vi=1.0/ni;
				for ( viter = pvect.begin() ; viter != pvect.end() ; viter++ ){
					i= *viter;
					//from j to i!!!
					M[i][j]=vi;
				}		
			}	
		}
	}

	//Product of M and r, rr[]=M[][]*r[]
	float mrprod(float *rr, float **M, float *r, int N){
		int i,j,tid;
		float sum=0;
		float maxdelta=0;
		//Parallel Computing
		#pragma omp parallel shared(M,r,rr,N) private(sum,i,j,tid)
		{
			//Parallel Computing
			#pragma omp for schedule(dynamic) nowait
			for(i=0;i<N;i++){
				tid = omp_get_thread_num();
				//printf("%d(%d) ",i,tid);
				sum=0;
				for (j=0;j<N;j++){
					sum+=M[i][j]*r[j];
					//printf("                 _Mij_=%f _rj_=%f _sum_=%f\n",M[i][j],r[j],sum);
				}
				//then calculate the new PageRank
				rr[i]=0.85*sum+0.15/N;
				//printf("rr[%d]=%e ",i+1,rr[i]);
			}
		}

		//Parallel Computing
		#pragma omp parallel// default(none) shared(r,rr) private(maxdelta,num,tid)
		{
			//Refresh the Page Rank SEPERATELY
			#pragma omp for reduction(max:maxdelta)
			for(int num=0;num<N;num++){
				tid = omp_get_thread_num();
				//printf("copy___tid=%d num=%d ",tid,num);
				if( maxdelta < fabs(rr[num]-r[num]) ){
					maxdelta = fabs(rr[num]-r[num]);
					//printf("maxdelta=%f\n",maxdelta);
				}
				r[num]=rr[num];
			}
		}
		return maxdelta;
	}


int main(){
	//hashmap of the graph
	hash_map< int, vector<int> > graphHash;
	graphHash=readFileToHashMap();
	
	//get the N
	int N=graphHash.size();
	
	//declair M[N][N]
	float **M;
	M = new float *[N];
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic) nowait
		for (int i = 0; i < N; i++)
		{
			M[i] = new float[N];
			for (int j = 0; j < N; j++){
				M[i][j] = 0.0;
			}
		}
	}

	//initialize the M[][]
	initM(graphHash,M);

	//initialize the r[]
	float r[N];
	for(int i=0; i<N; i++){
		r[i]=1.0/N;
	}
	float rr[N];

	//compute the PageRank iteratelly
	float delta=100000;
	float stop=0.000001e-8;
	int iterate=0;
	while(delta>stop){
		//iterate M*r once
		iterate++;
		delta=mrprod( rr, M, r, N);
		printf("=================NO.%d delta= %.16f\n",iterate,delta);
	}
	printf("Totally run %d times!\n",iterate);
	
	//initial the kvpair
	kvpairs outpair[N];
	for(int i=0; i<N; i++){
		outpair[i].id=i;
		outpair[i].pagerank=rr[i];
	}

	sort(outpair,outpair+N);

	//output file
	ofstream myfile;
	myfile.open ("Output_Task1.txt");
	int nout=0;
	while(nout<N){
		myfile.precision(20);//precision 20
		myfile.setf(ios::fixed); //floatfield is set to fixed
		myfile.unsetf(ios::scientific); //unset Scientific notation
		myfile<<outpair[nout].id<<" "<<outpair[nout].pagerank<<endl;
		nout++;
	}
	myfile.close();
	cout<<"finish!"<<endl;

	return 0;
}




