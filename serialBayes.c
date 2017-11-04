/*
    Program Bayes-Decision
    Kelompok 1
    CS-38-01
*/
#include<stdio.h>
#include<math.h>

#define ntraining 80
/*
typedef struct{
    float A1[ntraining];
    float A2[ntraining];
    char* kelas;
}training;
*/
float calculateSD(float data[],int n);
float bayesFunction(float mean1,float mean2,float stdev1,float stdev2,float A1,float A2);
int klasifikasi(float p1,float p2);

int main(){
//input data from file.
    FILE *f11,*f21;

    f11 = fopen("datatrainingA1.txt","r");
    f21 = fopen("datatrainingA2.txt","r");
    //f3 = fopen("kelastrainig.txt","r");

    int i;
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

    //mencetak data training yang sudah dibagi
    printf("setosaA1 \t setosaA2 \t versiA1 \t versiA2\n");
    while(i<nsetosatraining){
        printf("%1.2f \t\t %1.2f \t\t %1.2f \t\t %1.2f\n",trainingA1setosa[i],trainingA2setosa[i],trainingA1versi[i],trainingA2versi[i]);
        i++;
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


//menhitung mean dan var
    //menghitung mean
    //rata2 m01
    float sum01=0,sum02=0,m01,m02;
    while(i<nsetosatraining){
        sum01=sum01+trainingA1setosa[i];
        sum02=sum02+trainingA2setosa[i];
        i++;
    }
    m01=sum01/nsetosatraining;
    m02=sum02/nsetosatraining;
    //rata2 m02
    float sum11=0,sum12=0,m11,m12;
    while(j<nversitraining){
        sum11=sum11+trainingA1versi[j];
        sum12=sum12+trainingA2versi[j];
        j++;
    }
    m11 = sum11/nversitraining;
    m12 = sum12/nversitraining;

    printf("\n\n");
    printf("\nHasil Perhitungan mean\n");
    printf("m01 \t\t mo2 \n%f \t %f\n",m01,m02);
    printf("m11 \t\t m12 \n%f \t %f\n",m11,m12);

    //menghitung simpangan baku setiap klassnya
    printf("\nHasil Perhitungan STDEV\n");
    float s01,s02,s11,s12;
    s01=calculateSD(trainingA1setosa,nsetosatraining);
    s02=calculateSD(trainingA2setosa,nsetosatraining);
    s11=calculateSD(trainingA1versi,nversitraining);
    s12=calculateSD(trainingA2versi,nversitraining);

    printf("s01 \t\t s02\t\n");
    printf("%f \t %f\t\n",s01,s02);
    printf("s11 \t\t s12\t\n");
    printf("%f \t %f\t\n",s11,s12);
    printf("\n\n");

//menhitung peluang disetiap class
    //reset counter
    i=0;
    j=0;

    //inisiasi count
    int count=0;
    //hasil P setosa
    printf("===Hasil P Setosa===\n");
    printf("m&s_Asli \t m&s_Lawan \t Kondisi\n");
    while(i<ntestingsetosa){
        float p1=bayesFunction(m01,m02,s01,s02,testingA1setosa[i],testingA2setosa[i]);
        float p2=bayesFunction(m11,m12,s11,s12,testingA1setosa[i],testingA2setosa[i]);
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
        float p1=bayesFunction(m11,m12,s11,s12,testingA1versi[i],testingA2versi[i]);
        float p2=bayesFunction(m01,m02,s01,s02,testingA1versi[i],testingA2versi[i]);
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
    printf("Jumlah Data testing yang sesuai dengan kelasnya : %i\n",count);
    float akurasi=count*1.00/ntesting*1.00;
    printf("Akurasi : %1.02f '%'",akurasi);
//close

    return 0;
}

float calculateSD(float data[],int n)
{
    float sum = 0.0, mean, standardDeviation = 0.0;

    int i;

    for(i=0; i<n; ++i)
    {
        sum += data[i];
    }

    mean = sum/n;

    for(i=0; i<n; ++i){
        standardDeviation += pow(data[i] - mean, 2);
    }
    return sqrt(standardDeviation/n);
}

float bayesFunction(float mean1,float mean2,float stdev1,float stdev2,float A1,float A2){
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
