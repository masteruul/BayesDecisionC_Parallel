/*
    MPI Parallel Bayes Decision
    Kelompok 1 CS-38-01
*/
#include<stdio.h>
#include<math.h>
#include<mpi.h>
#include<stdlib.h>
#include<assert.h>


float hitung_rataan(float *array,int n);
float local_sum(float *array,int n);
float sqdiff(float *array,float mean,int n);
float bayes_function(float mean1,float mean2,float stdev1,float stdev2,float A1,float A2);
int klasifikasi(float p1,float p2);




int main(int argc,char** argv){

    //printf("go to Parallel Bayes (^^);
    //memanggil fungsi MPI_INIT
    MPI_Init(NULL,NULL);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    //printf("banyak Pemroses %i \n",world_size);

    if(argc != 1){
        fprintf(stderr,"Gunakan jumlah processor yang sesuai\n");
    }

    int root=0;

    FILE *f11,*f21;

        f11 = fopen("datatrainingA1.txt","r");
        f21 = fopen("datatrainingA2.txt","r");
        //f3 = fopen("kelastrainig.txt","r");

        int i,ntraining=80;
        float trainingA1[ntraining],trainingA2[ntraining];
        char* kelastraining[ntraining];

        i=0;
        //mengisikan data trining A1
        //printf("\t A1 \t\t A2 \n");
        while(i<ntraining){
            fscanf(f11,"%f ",&trainingA1[i]);
            fscanf(f21,"%f ",&trainingA2[i]);
            //printf("\t %f \t %f \n",trainingA1[i],trainingA2[i]);
            i++;
        }

 	// menginput data ke Array
        

        //pembagian data traininng ke kelasnya
        int nsetosatraining = 40,nversitraining = 40;
        float trainingA1setosa[nsetosatraining],trainingA1versi[nversitraining];
        float trainingA2setosa[nsetosatraining],trainingA2versi[nversitraining];

        int j=0;
        int v=0;
        while(j<ntraining){
            if(j<nsetosatraining){
                trainingA1setosa[j]=trainingA1[j];
                trainingA2setosa[j]=trainingA2[j];
            }else if(j>=nsetosatraining&&j<(nsetosatraining+nversitraining)){
                trainingA1versi[v]=trainingA1[j];
                trainingA2versi[v]=trainingA2[j];
                v++;
            }
            j++;
        }

	//reset counter
        i=0;
        j=0;

        //menginput data Testing
        FILE *f12,*f22;

        f12 = fopen("datatestingA1.txt","r");
        f22 = fopen("datatestingA2.txt","r");
        //f3 = fopen("kelastrainig.txt","r");

        int ntesting = 20,ntestingsetosa=10,ntestingversi=10;
        float testingA1[ntesting],testingA2[ntesting];
        char* kelastesting[ntesting];

        //mengisikan data trining A1
        //printf("\t A1 \t\t A2 \n");
        while(i<ntesting){
            fscanf(f12,"%f ",&testingA1[i]);
            fscanf(f22,"%f ",&testingA2[i]);
            //printf("\t %f \t %f \n",trainingA1[i],trainingA2[i]);
            i++;
        }

        float testingA1setosa[ntestingsetosa],testingA1versi[ntestingversi];
        float testingA2setosa[ntestingsetosa],testingA2versi[ntestingversi];


        int c=0;
        while(j<ntesting){
            if(j<ntestingsetosa){
                testingA1setosa[j]=testingA1[j];
                testingA2setosa[j]=testingA2[j];
            }else if(j>=ntestingsetosa&&j<(ntestingsetosa+ntestingversi)){
                testingA1versi[c]=testingA1[j];
                testingA2versi[c]=testingA2[j];
                c++;
            }
            j++;
        }

    if(world_rank==root){

       
        //reset counter
        i=0;
        j=0;

        //mencetak data training yang sudah dibagi
        printf("setosaA1 \t setosaA2 \t versiA1 \t versiA2\n");
        while(i<nsetosatraining){
            printf("%1.2f \t\t %1.2f \t\t %1.2f \t\t %1.2f\n",trainingA1setosa[i],trainingA2setosa[i],trainingA1versi[i],trainingA2versi[i]);
            i++;
        }
        

        //reset counter
        i=0;
        j=0;

        //mencetak data training yang sudah dibagi
        printf("\n=====Hasil Input Data Testing====\n");
        printf("setosaA1 \t setosaA2 \t versiA1 \t versiA2\n");
        while(i<ntestingsetosa&&i<ntestingversi){
            printf("%1.2f \t\t %1.2f \t\t %1.2f \t\t %1.2f\n",testingA1setosa[i],testingA2setosa[i],testingA1versi[i],testingA2versi[i]);
            i++;
        }



    }
        //sqiabel2 Parallel
        int data1_per_proc = 10;


    //masukkan data ke alamat data
    float *dataTrainingA1setosa=NULL;
    float *dataTrainingA2setosa=NULL;
    float *dataTrainingA1versi=NULL;
    float *dataTrainingA2versi=NULL;

    dataTrainingA1setosa=trainingA1setosa;
    dataTrainingA2setosa=trainingA2setosa;
    dataTrainingA1versi=trainingA1versi;
    dataTrainingA2versi=trainingA2versi;
    

    //membuat sqiabel sub data
    float *sub_dataTrainingA1setosa = (float*)malloc(sizeof(float)*data1_per_proc);
    float *sub_dataTrainingA2setosa = (float*)malloc(sizeof(float)*data1_per_proc);
    float *sub_dataTrainingA1versi = (float*)malloc(sizeof(float)*data1_per_proc);
    float *sub_dataTrainingA2versi = (float*)malloc(sizeof(float)*data1_per_proc);
    
    //memanggil MPI Scatter untuk membagi data Training dan mencari rata2 lokal
    MPI_Scatter(dataTrainingA1setosa,data1_per_proc,MPI_FLOAT,sub_dataTrainingA1setosa,data1_per_proc,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(dataTrainingA2setosa,data1_per_proc,MPI_FLOAT,sub_dataTrainingA2setosa,data1_per_proc,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(dataTrainingA1versi,data1_per_proc,MPI_FLOAT,sub_dataTrainingA1versi,data1_per_proc,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Scatter(dataTrainingA2versi,data1_per_proc,MPI_FLOAT,sub_dataTrainingA2versi,data1_per_proc,MPI_FLOAT,0,MPI_COMM_WORLD);

    float sub_rataanA1setosa=hitung_rataan(sub_dataTrainingA1setosa,data1_per_proc);
    float sub_rataanA2setosa=hitung_rataan(sub_dataTrainingA2setosa,data1_per_proc);
    float sub_rataanA1versi=hitung_rataan(sub_dataTrainingA1versi,data1_per_proc);
    float sub_rataanA2versi=hitung_rataan(sub_dataTrainingA2versi,data1_per_proc);

    float *sub_rataansA1setosa=NULL;
    float *sub_rataansA2setosa=NULL;
    float *sub_rataansA1versi=NULL;
    float *sub_rataansA2versi=NULL;

    if(world_rank==root){
        sub_rataansA1setosa=(float*)malloc(sizeof(float)*world_size);
		sub_rataansA2setosa=(float*)malloc(sizeof(float)*world_size);
        sub_rataansA1versi=(float*)malloc(sizeof(float)*world_size);
        sub_rataansA2versi=(float*)malloc(sizeof(float)*world_size);
                
	assert(sub_rataansA1setosa != NULL);
    	assert(sub_rataansA2setosa != NULL);
        assert(sub_rataansA1versi != NULL);
    	assert(sub_rataansA2versi != NULL);
    }
	
	//memanggin MPI_Gather untuk mencari nilai rataan_global
    MPI_Gather(&sub_rataanA1setosa,1,MPI_FLOAT,sub_rataansA1setosa,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(&sub_rataanA2setosa,1,MPI_FLOAT,sub_rataansA2setosa,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(&sub_rataanA1versi,1,MPI_FLOAT,sub_rataansA1versi,1,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Gather(&sub_rataanA2versi,1,MPI_FLOAT,sub_rataansA2versi,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    if(world_rank==root){
		float frataanA1setosa = hitung_rataan(sub_rataansA1setosa,world_size);
		float frataanA2setosa = hitung_rataan(sub_rataansA2setosa,world_size);
		float frataanA1versi = hitung_rataan(sub_rataansA1versi,world_size);
		float frataanA2versi = hitung_rataan(sub_rataansA2versi,world_size);
        
		printf("\n");
		printf("rataan dataTraining \n");
		printf("m01 \t\t m02 \t\n");
		printf("%f \t %f \n",frataanA1setosa,frataanA2setosa);
		printf("m11 \t\t m12 \t\n");
		printf("%f \t %f \n",frataanA1versi,frataanA2versi);
		
		

	/*
        float rataan_asliA1setosa=hitung_rataan(dataTrainingA1setosa,data1_per_proc*world_size);
        printf("rataan dataTraining Aslinya = %f\n",rataan_asliA1setosa);
	*/
    }
	

    //kode Program Paralel untuk mencari nilai standarDeviasi => Variansi
    //memanggil MPI Scatter untuk membagi data Training dan mencari rata2 lokal
    float local_sumA1setosa=local_sum(sub_dataTrainingA1setosa,data1_per_proc);
    float local_sumA2setosa=local_sum(sub_dataTrainingA2setosa,data1_per_proc);
    float local_sumA1versi=local_sum(sub_dataTrainingA1versi,data1_per_proc);
    float local_sumA2versi=local_sum(sub_dataTrainingA2versi,data1_per_proc);

    float global_sumA1setosa;
    float global_sumA2setosa;
    float global_sumA1versi;
    float global_sumA2versi;

    MPI_Allreduce(&local_sumA1setosa,&global_sumA1setosa,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&local_sumA2setosa,&global_sumA2setosa,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&local_sumA1versi,&global_sumA1versi,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&local_sumA2versi,&global_sumA2versi,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

		float meanA1setosa = global_sumA1setosa / (world_size*data1_per_proc);
        float meanA2setosa = global_sumA2setosa / (world_size*data1_per_proc);
        float meanA1versi = global_sumA1versi / (world_size*data1_per_proc);
        float meanA2versi = global_sumA2versi / (world_size*data1_per_proc);
	
/*
	printf("%f \t",local_sumA1setosa);
	printf("%f \t",global_sumA1setosa);
	printf("%f \n",meanA1setosa);
*/
	// ("%f \t",sqdiff(sub_dataTrainingA1setosa,meanA1setosa,data1_per_proc));
 	float sub_sqA1setosa=sqdiff(sub_dataTrainingA1setosa,meanA1setosa,data1_per_proc);
    float sub_sqA2setosa=sqdiff(sub_dataTrainingA2setosa,meanA2setosa,data1_per_proc);
   	float sub_sqA1versi=sqdiff(sub_dataTrainingA1versi,meanA1versi,data1_per_proc);
    float sub_sqA2versi=sqdiff(sub_dataTrainingA2versi,meanA2versi,data1_per_proc);
	
/* udah bisa
	printf("%f \t",sub_sqA1setosa);
	printf("%f \t\n",sub_sqA1versi);
*/
	float fsqA1setosa=0.0; 
	float fsqA2setosa=0.0;
	float fsqA1versi=0.0; 
	float fsqA2versi=0.0; 

	MPI_Reduce(&sub_sqA1setosa,&fsqA1setosa,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&sub_sqA2setosa,&fsqA2setosa,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&sub_sqA1versi,&fsqA1versi,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&sub_sqA2versi,&fsqA2versi,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

	
	float stdev01 = sqrt(fsqA1setosa/nsetosatraining);
	float stdev02 = sqrt(fsqA2setosa/nsetosatraining);
	float stdev11 = sqrt(fsqA1versi/nversitraining);
	float stdev12 = sqrt(fsqA2versi/nversitraining);
	
    if(world_rank==root){
        printf("\n");
	    printf("StandarDeviasi dataTraining \n");
		printf("s01 \t\t s02 \t\n");
		printf("%f \t %f \n",stdev01,stdev02);
		printf("s11 \t\t s12 \t\n");
		printf("%f \t %f \n",stdev11,stdev12);

	/*
        float rataan_asliA1setosa=hitung_rataan(dataTrainingA1setosa,data1_per_proc*world_size);

        printf("rataan dataTraining Aslinya = %f\n",rataan_asliA1setosa);
	*/
		
    }
	
	//pengisian variabel m01,m02,m03,m04 untuk BayesFunction
		float m01=meanA1setosa;
		float m02=meanA2setosa;
		float m11=meanA1versi;
		float m12=meanA2versi;
		
	//mengisi nilai s01,s02,s11,s12 untuk mengoperasikan BayesFunction
		float s01 = stdev01;
		float s02 = stdev02;
		float s11 = stdev11;
		float s12 = stdev12;
		
	
	
	//kode untuk Fungsi bayes_function (serial melalui node root)
	if(world_rank==root){
		//reset counter
		i=0;
		j=0;

		//inisiasi count
		int count=0;
		//hasil P setosa
		printf("\n");
		printf("===Hasil P Setosa===\n");
		printf("m&s_Asli \t m&s_Lawan \t Kondisi\n");
		while(i<ntestingsetosa){
			float p1=bayes_function(m01,m02,s01,s02,testingA1setosa[i],testingA2setosa[i]);
			float p2=bayes_function(m11,m12,s11,s12,testingA1setosa[i],testingA2setosa[i]);
			printf("%f",p1);
			printf("\t %f",p2);
			if(klasifikasi(p1,p2)==1){
				printf("\t true \n");
				count++;
			}else{
				printf("\t false \n");
			}

			i++;
		}
		
		 //reset counter
		i=0;
		j=0;
		printf("\n\n");
		//hasil P versi
		printf("===Hasil P versi===\n");
		printf("m&s_Asli \t m&s_Lawan\n");
		while(i<ntestingversi){
			float p1=bayes_function(m11,m12,s11,s12,testingA1versi[i],testingA2versi[i]);
			float p2=bayes_function(m01,m02,s01,s02,testingA1versi[i],testingA2versi[i]);
			printf("%f",p1);
			printf("\t %f",p2);
			if(klasifikasi(p1,p2)==1){
				printf("\t true \n");
				count++;
			}else{
				printf("\t false \n");
			}
			i++;
		}
		//
		//output data
		//Akurasi
		//reset counter
		printf("\n");
		printf("Jumlah Data testing yang sesuai dengan kelasnya : %i\n",count);
		float akurasi=count*1.00/ntesting*1.00;
		printf("Akurasi : %1.02f",akurasi);
		printf("%\n");
		
	}
	


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
	
	return 0;
}

float hitung_rataan(float *array,int n){
    float sum = 0.f;
    int i;
    for(i=0;i<n;i++){
        sum=sum+array[i];
    }

    return sum/n;
}

float local_sum(float *array,int n){
    float sum=0.0;
    int i;
    for(i=0;i<n;i++){
    	sum +=array[i];
    }
    return sum; 
}

float sqdiff(float *array,float mean,int n){
    float sq_diff=0.0;
    int i;
    for(i=0;i<n;i++){
    	sq_diff = sq_diff+(array[i]-mean)*(array[i]-mean);
    }
    return sq_diff;		
}

float bayes_function(float mean1,float mean2,float stdev1,float stdev2,float A1,float A2){
	
	float P=0.0;
    float satuperexp1 =1/(2*3.14*pow(stdev1,2));
    float satuperexp2 =1/(2*3.14*pow(stdev2,2));
    float temp1,temp2;
    //printf("%f\n",satuperexp);
    temp1=exp(-1*(pow((A1-mean1),2)/(2*pow(stdev1,2))));
    temp2=exp(-1*(pow((A2-mean2),2)/(2*pow(stdev2,2))));
    P=satuperexp1*temp1*satuperexp2*temp2;
    return P;
    //return P;
	
}

int klasifikasi(float p1,float p2){
    if(p1>p2){
        return 1;
    }else{
        return 0;
    }
}
