//bayesian_matting
/**
* @file recursive_gaussian.cpp
* @author Sergey Smirnov <sergey.smirnov@tut.fi>
* @date 29.10.2014
* @copyright 3D Media Group / Tampere University of Technology
*/



#ifdef __cplusplus 
extern "C" {
#include "mex.h"
}
#endif // __cplusplus

#include "../common/meximage.h"
//#include "../common/defines.h"
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>
#include <vector>
#ifndef _DEBUG
#include <omp.h>
#endif

//#define M_PI       3.14159265358979323846

#pragma comment (lib, "libeng.lib") 
#pragma comment (lib, "libmx.lib")
#pragma comment (lib, "libmex.lib")

#define colors 3

//typedef unsigned char uint8;
using namespace mymex;
using namespace std;

#define BACKGROUND_VALUE            0
#define FOREGROUND_VALUE          255

#define BAYESIAN_NUMBER_NEAREST   200
#define BAYESIAN_SIGMA            8.f
#define BAYESIAN_SIGMA_C          5.f
#define BAYESIAN_MAX_CLUS           3

struct CvPoint
{
	float x;
	float y;
};

typedef MexImage<float> IplImage;
typedef float CvMat;

//Implementation of	"A Bayesian Approach to Digital Matting"
class BayesianMatting
{
private:
	const int width;
	const int height;
	const long HW; 
	MexImage<uint8_t> *colorImg;
	MexImage<uint8_t> *fgImg, *bgImg;
	MexImage<uint8_t> *bgmask, *fgmask, *unmask, *unsolvedmask;
public:

	MexImage<uint8_t> *trimap;
	MexImage<float> *alphamap;

	//the input is the color image and the trimap image
	//the format is 3 channels + uchar and 1 channel + uchar respectively
	BayesianMatting(MexImage<uint8_t>* _cImg, MexImage<uint8_t>* _trimap) : trimap(_trimap), colorImg(_cImg), width(_cImg->width), height(_cImg->height), HW(_cImg->width*_cImg->height)
	{
		//Initialize();
		fgImg = new MexImage<uint8_t>(width, height, 3);// cvCreateImage(cvGetSize(trimap), 8, 3);
		bgImg = new MexImage<uint8_t>(width, height, 3);// = cvCreateImage(cvGetSize(trimap), 8, 3);
		fgmask = new MexImage<uint8_t>(width, height, 1);// = cvCreateImage(cvGetSize(trimap), 8, 1);
		bgmask = new MexImage<uint8_t>(width, height, 1);// = cvCreateImage(cvGetSize(trimap), 8, 1);
		unmask = new MexImage<uint8_t>(width, height, 1);// = cvCreateImage(cvGetSize(trimap), 8, 1);
		unsolvedmask = new MexImage<uint8_t>(width, height, 1);// = cvCreateImage(cvGetSize(trimap), 8, 1);
		alphamap = new MexImage<float>(width, height, 1);// = cvCreateImage(cvGetSize(colorImg), 32, 1);

		fgImg->setval(0); //cvZero(fgImg);
		bgImg->setval(0);  //cvZero(bgImg);
		//fgmask->setval(0);  //cvZero(fgmask);
		//bgmask->setval(0);  //cvZero(bgmask);
		//unmask->setval(0);  //cvZero(unmask);
		//alphamap->setval(0); //cvZero(alphamap);
		//cvZero( unsolvedmask );	
		
		for (long i = 0; i < HW; i++)
		{			
			alphamap->at(i) = float(trimap->at(i) == FOREGROUND_VALUE);
			if (trimap->at(i) == BACKGROUND_VALUE)
			{
				bgmask->at(i) = 255;
				fgmask->at(i) = 0;
				unmask->at(i) = 0;
				unsolvedmask->at(i) = 0;
				
				for (int c = 0; c < colors; c++)
				{
					bgImg->at(c*HW + i) = colorImg->at(c*HW + i);
				}
			}
			else if (trimap->at(i) == FOREGROUND_VALUE)
			{
				bgmask->at(i) = 0;
				fgmask->at(i) = 255;
				unmask->at(i) = 0;
				unsolvedmask->at(i) = 0;
				for (int c = 0; c < colors; c++)
				{
					fgImg->at(c*HW + i) = colorImg->at(c*HW + i);
				}
			}
			else
			{
				bgmask->at(i) = 0;
				fgmask->at(i) = 0;
				unmask->at(i) = 255;
				unsolvedmask->at(i) = 255;
			}
		}			

		SetParameter();
	}

	~BayesianMatting()
	{
		delete fgImg;
		delete bgImg;
		delete fgmask;
		delete bgmask;
		delete unmask;
		delete unsolvedmask;
		delete alphamap;
	}

	//void Initialize();

	//set parameter
	void SetParameter(int N = BAYESIAN_NUMBER_NEAREST, float sigma_ = BAYESIAN_SIGMA, float sigma_c = BAYESIAN_SIGMA_C)
	{
		nearest = N;
		sigmac = sigma_c;
		sigma = sigma_;
	}

	//solve the matting problem
	double Solve();



private:
	/* ==================================================================================
	Internal functions.
	================================================================================== */
	//get the extreme outer contours of an image
	void GetContour(MexImage<uint8_t>* img, vector<CvPoint> &contour);

	//initialize the alpha of one point using the mean of neighors and save the result in alphamap
	void InitializeAlpha(int r, int c, const IplImage* unSolvedMask);

	//used for clustering according the equation in paper "Color Quantization of Images"
	void CalculateNonNormalizeCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, CvMat* mean, CvMat* cov);

	// calculate mean and cov of the given clus_set
	void CalculateMeanCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, CvMat* mean, CvMat* cov);

	// calculate weight, mean and cov of the given clus_set
	void CalculateWeightMeanCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, float &weight, CvMat* mean, CvMat* cov);

	//get the foreground and backgroud gmm model at a given pixel
	void GetGMMModel(int r, int c, vector<float> &fg_weight, const vector<CvMat*> fg_mean, const vector<CvMat*> inv_fg_cov, vector<float> &bg_weight, const vector<CvMat*> bg_mean, const vector<CvMat*> inv_bg_cov);

	//collect the foreground/background sample set from the contour, called by GetGMMModel
	void CollectSampleSet(int r, int c, vector<pair<CvPoint, float>> &fg_set, vector<pair<CvPoint, float>> &bg_set);

	//solve Eq. (9) at pixel (r,c) according to the alpha in alphamap and save the result in fgImg and bgImg
	void SolveBF(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov);

	//solve Eq. (10) at pixel (r,c) according to the foreground and background color in fgImg and bgImg, and save the result in alphamap
	inline void SolveAlpha(int r, int c);

	//compute total likelihood
	float computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov);

	float computeLikelihood(int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov);


	/* ==================================================================================
	Offline Variables.
	================================================================================== */
	int   nearest;
	float sigma;
	float sigmac;


};


double BayesianMatting::Solve()
{
	int p, r, c, i, j, iter, fgClus, bgClus;
	int outter;
	float L, maxL;

	IplImage* shownImg = cvCreateImage(cvGetSize(colorImg), 8, 3);
	IplImage* solveAgainMask = cvCreateImage(cvGetSize(unmask), 8, 1);

	vector<float>   fg_weight(BAYESIAN_MAX_CLUS, 0);
	vector<float>   bg_weight(BAYESIAN_MAX_CLUS, 0);
	vector<CvMat *> fg_mean(BAYESIAN_MAX_CLUS, NULL);
	vector<CvMat *> bg_mean(BAYESIAN_MAX_CLUS, NULL);
	vector<CvMat *> inv_fg_cov(BAYESIAN_MAX_CLUS, NULL);
	vector<CvMat *> inv_bg_cov(BAYESIAN_MAX_CLUS, NULL);
	for (i = 0; i<BAYESIAN_MAX_CLUS; i++)
	{
		fg_mean[i] = cvCreateMat(3, 1, CV_32FC1);
		bg_mean[i] = cvCreateMat(3, 1, CV_32FC1);
		inv_fg_cov[i] = cvCreateMat(3, 3, CV_32FC1);
		inv_bg_cov[i] = cvCreateMat(3, 3, CV_32FC1);
	}

	for (int Iteration = 0; Iteration<1; ++Iteration)
	{
		printf("\niteration %d:\n", Iteration);

		if (Iteration)
			cvCopy(unmask, solveAgainMask);

		outter = 0;
		for (;;)
		{
			printf("solving contour %d\r", outter++);

			vector<CvPoint> toSolveList;

			if (!Iteration)
				GetContour(unsolvedmask, toSolveList);
			else
				GetContour(solveAgainMask, toSolveList);

			//no unknown left
			if (!toSolveList.size())
				break;

			cvCopyImage(colorImg, shownImg);
			for (int k = 0; k<toSolveList.size(); ++k)
				cvCircle(shownImg, toSolveList[k], 1, cvScalarAll(128));
			cvNamedWindow("points to solve");
			cvShowImage("points to solve", shownImg);
			cvMoveWindow("points to solve", 0, 0);
			cvWaitKey(1);


			//solve the points in the list one by one
			for (p = 0; p < toSolveList.size(); ++p)
			{
				r = toSolveList[p].y, c = toSolveList[p].x;

				//get the gmm model using the neighbors of foreground and neighbors of background			
				GetGMMModel(r, c, fg_weight, fg_mean, inv_fg_cov, bg_weight, bg_mean, inv_bg_cov);

				maxL = (float)-INT_MAX;

				for (i = 0; i<BAYESIAN_MAX_CLUS; i++)
					for (j = 0; j<BAYESIAN_MAX_CLUS; j++)
					{
						//initilize the alpha by the average of near points
						if (!Iteration)
							InitializeAlpha(r, c, unsolvedmask);
						else
							InitializeAlpha(r, c, solveAgainMask);

						for (iter = 0; iter<3; ++iter)
						{
							SolveBF(r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j]);
							SolveAlpha(r, c);
						}

						// largest likelihood, restore the index in fgClus, bgClus
						L = computeLikelihood(r, c, fg_mean[i], inv_fg_cov[i], bg_mean[j], inv_bg_cov[j]);
						//L = computeLikelihood( r, c, fg_weight[i], fg_mean[i], inv_fg_cov[i], bg_weight[j], bg_mean[j], inv_bg_cov[j]);
						if (L>maxL)
						{
							maxL = L;
							fgClus = i;
							bgClus = j;
						}
					}


				if (!Iteration)
					InitializeAlpha(r, c, unsolvedmask);
				else
					InitializeAlpha(r, c, solveAgainMask);

				for (iter = 0; iter<5; ++iter)
				{
					SolveBF(r, c, fg_mean[fgClus], inv_fg_cov[fgClus], bg_mean[bgClus], inv_bg_cov[bgClus]);
					SolveAlpha(r, c);
				}
				//printf("%f\n", CV_IMAGE_ELEM(alphamap,float,r,c));

				//solved!
				if (!Iteration)
					CV_IMAGE_ELEM(unsolvedmask, uchar, r, c) = 0;
				else
					CV_IMAGE_ELEM(solveAgainMask, uchar, r, c) = 0;
			}
			//cvNamedWindow("fg");
			//cvShowImage("fg", fgImg );
			//cvMoveWindow("fg",0,100+colorImg->height);
			//cvNamedWindow("bg");
			//cvShowImage("bg", bgImg );
			//cvMoveWindow("bg",100+colorImg->width,100+colorImg->height);
			cvNamedWindow("alphamap");
			cvShowImage("alphamap", alphamap);
			cvMoveWindow("alphamap", 100 + colorImg->width, 0);
			cvWaitKey(1);
		}
	}

	printf("\nDone!!\n");

	/////////////////////////

	cvReleaseImage(&shownImg);
	cvReleaseImage(&solveAgainMask);

	for (i = 0; i<fg_mean.size(); i++)
	{
		cvReleaseMat(&fg_mean[i]);
		cvReleaseMat(&bg_mean[i]);
		cvReleaseMat(&inv_fg_cov[i]);
		cvReleaseMat(&inv_bg_cov[i]);
	}
	return 1;
}

void BayesianMatting::InitializeAlpha(int r, int c, const IplImage* unSolvedMask)
{
	int i, j;
	int min_x, min_y, max_x, max_y;
#define WIN_SIZE 1


	min_x = max(0, c - WIN_SIZE);
	min_y = max(0, r - WIN_SIZE);
	max_x = min(colorImg->width - 1, c + WIN_SIZE);
	max_y = min(colorImg->height - 1, r + WIN_SIZE);

	int count = 0;
	float sum = 0;
	for (i = min_y; i <= max_y; ++i)
		for (j = min_x; j <= max_x; ++j)
		{
			if (!CV_IMAGE_ELEM(unSolvedMask, uchar, i, j))
			{
				sum += CV_IMAGE_ELEM(alphamap, float, i, j);
				++count;
			}
		}

	CV_IMAGE_ELEM(alphamap, float, r, c) = (count ? sum / count : 0);
}

void BayesianMatting::GetContour(IplImage* img, vector<CvPoint> &contour)
{
	contour.clear();

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contours = 0;

	cvFindContours(img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	for (; contours != 0; contours = contours->h_next)
	{
		CvSeqReader reader;
		cvStartReadSeq(contours, &reader, 0);

		int i, count = contours->total;

		CvPoint pt;
		for (i = 0; i < count; i++)
		{
			CV_READ_SEQ_ELEM(pt, reader);
			contour.push_back(cvPoint(pt.x, pt.y));
		}
	}

	cvReleaseMemStorage(&storage);
}

void BayesianMatting::CalculateNonNormalizeCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, CvMat* mean, CvMat* cov)
{
	int cur_r, cur_c;
	float cur_w, total_w = 0;
	cvZero(mean);
	cvZero(cov);
	for (size_t j = 0; j<clus_set.size(); j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for (int h = 0; h<3; h++)
		{
			CV_MAT_ELEM(*mean, float, h, 0) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h));
			for (int k = 0; k<3; k++)
				CV_MAT_ELEM(*cov, float, h, k) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h)*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + k));
		}

		total_w += clus_set[j].second;
	}

	float inv_total_w = 1.f / total_w;
	for (int h = 0; h<3; h++)
		for (int k = 0; k<3; k++)
			CV_MAT_ELEM(*cov, float, h, k) -= (inv_total_w*CV_MAT_ELEM(*mean, float, h, 0)*CV_MAT_ELEM(*mean, float, k, 0));

}

void BayesianMatting::CalculateMeanCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, CvMat* mean, CvMat* cov)
{
	int cur_r, cur_c;
	float cur_w, total_w = 0;
	cvZero(mean);
	cvZero(cov);
	for (size_t j = 0; j<clus_set.size(); j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for (int h = 0; h<3; h++)
		{
			CV_MAT_ELEM(*mean, float, h, 0) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h));
			for (int k = 0; k<3; k++)
				CV_MAT_ELEM(*cov, float, h, k) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h)*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + k));
		}

		total_w += clus_set[j].second;
	}

	float inv_total_w = 1.f / total_w;
	for (int h = 0; h<3; h++)
	{
		CV_MAT_ELEM(*mean, float, h, 0) *= inv_total_w;
		for (int k = 0; k<3; k++)
			CV_MAT_ELEM(*cov, float, h, k) *= inv_total_w;
	}

	for (int h = 0; h<3; h++)
		for (int k = 0; k<3; k++)
			CV_MAT_ELEM(*cov, float, h, k) -= (CV_MAT_ELEM(*mean, float, h, 0)*CV_MAT_ELEM(*mean, float, k, 0));
}

void BayesianMatting::CalculateWeightMeanCov(const IplImage *cImg, const vector<pair<CvPoint, float>> &clus_set, float &weight, CvMat* mean, CvMat* cov)
{
	int cur_r, cur_c;
	float cur_w, total_w = 0;
	cvZero(mean);
	cvZero(cov);
	for (size_t j = 0; j<clus_set.size(); j++)
	{
		cur_r = clus_set[j].first.y;
		cur_c = clus_set[j].first.x;
		cur_w = clus_set[j].second;
		for (int h = 0; h<3; h++)
		{
			CV_MAT_ELEM(*mean, float, h, 0) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h));
			for (int k = 0; k<3; k++)
				CV_MAT_ELEM(*cov, float, h, k) += (cur_w*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + h)*CV_IMAGE_ELEM(cImg, uchar, cur_r, 3 * cur_c + k));
		}

		total_w += clus_set[j].second;
	}

	float inv_total_w = 1.f / total_w;
	for (int h = 0; h<3; h++)
	{
		CV_MAT_ELEM(*mean, float, h, 0) *= inv_total_w;
		for (int k = 0; k<3; k++)
			CV_MAT_ELEM(*cov, float, h, k) *= inv_total_w;
	}

	for (int h = 0; h<3; h++)
		for (int k = 0; k<3; k++)
			CV_MAT_ELEM(*cov, float, h, k) -= (CV_MAT_ELEM(*mean, float, h, 0)*CV_MAT_ELEM(*mean, float, k, 0));

	weight = total_w;
}


void BayesianMatting::GetGMMModel(int r, int c, vector<float> &fg_weight, const vector<CvMat*> fg_mean, const vector<CvMat*> inv_fg_cov, vector<float> &bg_weight, const vector<CvMat*> bg_mean, const vector<CvMat*> inv_bg_cov)
{
	vector<pair<CvPoint, float>> fg_set, bg_set;
	CollectSampleSet(r, c, fg_set, bg_set);

	//IplImage* tmp1 = cvCloneImage( colorImg );	
	//IplImage* tmp2 = cvCloneImage( colorImg );	
	//			
	//for(size_t i=0;i<fg_set.size();++i)
	//{
	//	cvCircle( tmp1, fg_set[i].first, 1, cvScalar( 0, 0, fg_set[i].second * 255 ) );	
	//	cvCircle( tmp2, bg_set[i].first, 1, cvScalar( bg_set[i].second * 255, 0, 0 ) );	
	//}

	//cvNamedWindow( "fg_sample" );
	//cvShowImage( "fg_sample", tmp1 );
	//cvNamedWindow( "bg_sample" );
	//cvShowImage( "bg_sample", tmp2 );
	//cvWaitKey( 0 );
	//cvReleaseImage( &tmp1 );	
	//cvReleaseImage( &tmp2 );	

	CvMat *mean = cvCreateMat(3, 1, CV_32FC1);
	CvMat *cov = cvCreateMat(3, 3, CV_32FC1);
	CvMat *inv_cov = cvCreateMat(3, 3, CV_32FC1);
	CvMat *eigval = cvCreateMat(3, 1, CV_32FC1);
	CvMat *eigvec = cvCreateMat(3, 3, CV_32FC1);
	CvMat *cur_color = cvCreateMat(3, 1, CV_32FC1);
	CvMat *max_eigvec = cvCreateMat(3, 1, CV_32FC1);
	CvMat *target_color = cvCreateMat(3, 1, CV_32FC1);
	//fg

	//// initializtion
	vector<pair<CvPoint, float>> clus_set[BAYESIAN_MAX_CLUS];
	int nClus = 1;
	clus_set[0] = fg_set;

	while (nClus<BAYESIAN_MAX_CLUS)
	{
		// find the largest eigenvalue
		double max_eigval = 0;
		int max_idx = 0;
		for (int i = 0; i<nClus; i++)
		{
			//CalculateMeanCov(clus_set[i],mean,cov);
			CalculateNonNormalizeCov(fgImg, clus_set[i], mean, cov);

			// compute eigval, and eigvec
			cvSVD(cov, eigval, eigvec);
			if (cvmGet(eigval, 0, 0)>max_eigval)
			{
				cvGetCol(eigvec, max_eigvec, 0);
				max_eigval = cvmGet(eigval, 0, 0);
				max_idx = i;
			}
		}

		// split
		vector<pair<CvPoint, float>> new_clus_set[2];
		CalculateMeanCov(fgImg, clus_set[max_idx], mean, cov);
		double boundary = cvDotProduct(mean, max_eigvec);
		for (size_t i = 0; i<clus_set[max_idx].size(); i++)
		{
			for (int j = 0; j<3; j++)
				cvmSet(cur_color, j, 0, CV_IMAGE_ELEM(fgImg, uchar, clus_set[max_idx][i].first.y, 3 * clus_set[max_idx][i].first.x + j));

			if (cvDotProduct(cur_color, max_eigvec)>boundary)
				new_clus_set[0].push_back(clus_set[max_idx][i]);
			else
				new_clus_set[1].push_back(clus_set[max_idx][i]);
		}

		clus_set[max_idx] = new_clus_set[0];
		clus_set[nClus] = new_clus_set[1];

		nClus += 1;
	}

	// return all the mean and cov of fg
	float weight_sum, inv_weight_sum;
	weight_sum = 0;
	for (int i = 0; i<nClus; i++)
	{
		CalculateWeightMeanCov(fgImg, clus_set[i], fg_weight[i], fg_mean[i], cov);
		cvInvert(cov, inv_fg_cov[i]);
		weight_sum += fg_weight[i];
	}
	//normalize weight
	inv_weight_sum = 1.f / weight_sum;
	for (int i = 0; i<nClus; i++)
		fg_weight[i] *= inv_weight_sum;

	// bg
	// initializtion
	nClus = 1;
	for (int i = 0; i<BAYESIAN_MAX_CLUS; ++i)
		clus_set[i].clear();
	clus_set[0] = bg_set;

	while (nClus<BAYESIAN_MAX_CLUS)
	{
		// find the largest eigenvalue
		double max_eigval = 0;
		int max_idx = 0;
		for (int i = 0; i<nClus; i++)
		{
			//CalculateMeanCov(clus_set[i],mean,cov);
			CalculateNonNormalizeCov(bgImg, clus_set[i], mean, cov);

			// compute eigval, and eigvec
			cvSVD(cov, eigval, eigvec);
			if (cvmGet(eigval, 0, 0)>max_eigval)
			{
				cvGetCol(eigvec, max_eigvec, 0);
				max_eigval = cvmGet(eigval, 0, 0);
				max_idx = i;
			}
		}

		// split
		vector<pair<CvPoint, float>> new_clus_set[2];
		CalculateMeanCov(bgImg, clus_set[max_idx], mean, cov);
		double boundary = cvDotProduct(mean, max_eigvec);
		for (size_t i = 0; i<clus_set[max_idx].size(); i++)
		{
			for (int j = 0; j<3; j++)
				cvmSet(cur_color, j, 0, CV_IMAGE_ELEM(bgImg, uchar, clus_set[max_idx][i].first.y, 3 * clus_set[max_idx][i].first.x + j));

			if (cvDotProduct(cur_color, max_eigvec)>boundary)
				new_clus_set[0].push_back(clus_set[max_idx][i]);
			else
				new_clus_set[1].push_back(clus_set[max_idx][i]);
		}

		clus_set[max_idx] = new_clus_set[0];
		clus_set[nClus] = new_clus_set[1];

		nClus += 1;
	}

	// return all the mean and cov of bg
	weight_sum = 0;
	for (int i = 0; i<nClus; i++)
	{
		CalculateWeightMeanCov(bgImg, clus_set[i], bg_weight[i], bg_mean[i], cov);
		cvInvert(cov, inv_bg_cov[i]);
		weight_sum += bg_weight[i];
	}
	//normalize weight
	inv_weight_sum = 1.f / weight_sum;
	for (int i = 0; i<nClus; i++)
		bg_weight[i] *= inv_weight_sum;

	cvReleaseMat(&mean), cvReleaseMat(&cov), cvReleaseMat(&eigval), cvReleaseMat(&eigvec), cvReleaseMat(&cur_color), cvReleaseMat(&inv_cov), cvReleaseMat(&max_eigvec), cvReleaseMat(&target_color);
}

void BayesianMatting::CollectSampleSet(int r, int c, vector<pair<CvPoint, float>> &fg_set, vector<pair<CvPoint, float>> &bg_set)
{
	fg_set.clear(), bg_set.clear();
#define UNSURE_DISTANCE 1	

	pair<CvPoint, float> sample;
	float dist_weight;
	float inv_2sigma_square = 1.f / (2 * sigma*sigma);

	int dist = 1;
	while (fg_set.size() < nearest)
	{
		if (r - dist >= 0)
		{
			for (int z = max(0, c - dist); z <= min(colorImg->width - 1, c + dist); ++z)
			{
				dist_weight = expf(-(dist*dist + (z - c)*(z - c)) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(fgmask, uchar, r - dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second = dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r - dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r - dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r - dist, z) * CV_IMAGE_ELEM(alphamap, float, r - dist, z) * dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if (r + dist < colorImg->height)
		{
			for (int z = max(0, c - dist); z <= min(colorImg->width - 1, c + dist); ++z)
			{
				dist_weight = expf(-(dist*dist + (z - c)*(z - c)) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(fgmask, uchar, r + dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second = dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r + dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r + dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second = CV_IMAGE_ELEM(alphamap, float, r + dist, z) * CV_IMAGE_ELEM(alphamap, float, r + dist, z) * dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if (c - dist >= 0)
		{
			for (int z = max(0, r - dist + 1); z <= min(colorImg->height - 1, r + dist - 1); ++z)
			{
				dist_weight = expf(-((z - r)*(z - r) + dist*dist) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(fgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c - dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c - dist) * CV_IMAGE_ELEM(alphamap, float, z, c - dist) * dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		if (c + dist < colorImg->width)
		{
			for (int z = max(0, r - dist + 1); z <= min(colorImg->height - 1, r + dist - 1); ++z)
			{
				dist_weight = expf(-((z - r)*(z - r) + dist*dist) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(fgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c + dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = CV_IMAGE_ELEM(alphamap, float, z, c + dist) * CV_IMAGE_ELEM(alphamap, float, z, c + dist) * dist_weight;

					fg_set.push_back(sample);
					if (fg_set.size() == nearest)
						goto BG;
				}
			}
		}

		++dist;
	}

BG:
	int bg_unsure = 0;
	dist = 1;

	while (bg_set.size() < nearest)
	{
		dist_weight = expf(-(dist*dist) / (2 * sigma*sigma));
		if (r - dist >= 0)
		{
			for (int z = max(0, c - dist); z <= min(colorImg->width - 1, c + dist); ++z)
			{
				dist_weight = expf(-(dist*dist + (z - c)*(z - c)) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(bgmask, uchar, r - dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second = dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r - dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r - dist, z))
				{
					sample.first.x = z;
					sample.first.y = r - dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r - dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r - dist, z)) * dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if (r + dist < colorImg->height)
		{
			for (int z = max(0, c - dist); z <= min(colorImg->width - 1, c + dist); ++z)
			{
				dist_weight = expf(-(dist*dist + (z - c)*(z - c)) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(bgmask, uchar, r + dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second = dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, r + dist, z) && !CV_IMAGE_ELEM(unsolvedmask, uchar, r + dist, z))
				{
					sample.first.x = z;
					sample.first.y = r + dist;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, r + dist, z)) * (1 - CV_IMAGE_ELEM(alphamap, float, r + dist, z)) * dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if (c - dist >= 0)
		{
			for (int z = max(0, r - dist + 1); z <= min(colorImg->height - 1, r + dist - 1); ++z)
			{
				dist_weight = expf(-((z - r)*(z - r) + dist*dist) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(bgmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c - dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c - dist))
				{
					sample.first.x = c - dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c - dist)) * dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		if (c + dist < colorImg->width)
		{
			for (int z = max(0, r - dist + 1); z <= min(colorImg->height - 1, r + dist - 1); ++z)
			{
				dist_weight = expf(-((z - r)*(z - r) + dist*dist) * inv_2sigma_square);

				if (CV_IMAGE_ELEM(bgmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
				else if (dist < UNSURE_DISTANCE && CV_IMAGE_ELEM(unmask, uchar, z, c + dist) && !CV_IMAGE_ELEM(unsolvedmask, uchar, z, c + dist))
				{
					sample.first.x = c + dist;
					sample.first.y = z;
					sample.second = (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * (1 - CV_IMAGE_ELEM(alphamap, float, z, c + dist)) * dist_weight;

					bg_set.push_back(sample);
					if (bg_set.size() == nearest)
						goto DONE;
				}
			}
		}

		++dist;
	}

DONE:
	assert(fg_set.size() == nearest);
	assert(bg_set.size() == nearest);
}

void BayesianMatting::SolveBF(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov)
{
	CvMat *A = cvCreateMat(6, 6, CV_32FC1);
	CvMat *x = cvCreateMat(6, 1, CV_32FC1);
	CvMat *b = cvCreateMat(6, 1, CV_32FC1);
	CvMat *I = cvCreateMat(3, 3, CV_32FC1);
	CvMat *work_3x3 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *work_3x1 = cvCreateMat(3, 1, CV_32FC1);

	float alpha = CV_IMAGE_ELEM(alphamap, float, r, c);
	CvScalar fg_color = cvScalar(CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c), CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 1), CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 2));
	CvScalar bg_color = cvScalar(CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c), CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1), CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2));
	CvScalar  c_color = cvScalar(CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c), CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c + 1), CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c + 2));

	float inv_sigmac_square = 1.f / (sigmac*sigmac);

	cvZero(I);
	CV_MAT_ELEM(*I, float, 0, 0) = CV_MAT_ELEM(*I, float, 1, 1) = CV_MAT_ELEM(*I, float, 2, 2) = 1.f;

	////a
	cvCvtScale(I, work_3x3, alpha*alpha*inv_sigmac_square);
	cvAdd(inv_fg_cov, work_3x3, work_3x3);
	for (int i = 0; i<3; ++i)
		for (int j = 0; j<3; ++j)
			CV_MAT_ELEM(*A, float, i, j) = CV_MAT_ELEM(*work_3x3, float, i, j);

	//
	cvCvtScale(I, work_3x3, alpha*(1 - alpha)*inv_sigmac_square);
	for (int i = 0; i<3; ++i)
		for (int j = 0; j<3; ++j)
			CV_MAT_ELEM(*A, float, i, 3 + j) = CV_MAT_ELEM(*A, float, 3 + i, j) = CV_MAT_ELEM(*work_3x3, float, i, j);

	//
	cvCvtScale(I, work_3x3, (1 - alpha)*(1 - alpha)*inv_sigmac_square);
	cvAdd(inv_bg_cov, work_3x3, work_3x3);
	for (int i = 0; i<3; ++i)
		for (int j = 0; j<3; ++j)
			CV_MAT_ELEM(*A, float, 3 + i, 3 + j) = CV_MAT_ELEM(*work_3x3, float, i, j);

	////x
	cvZero(x);

	////b
	cvMatMul(inv_fg_cov, fg_mean, work_3x1);
	for (int i = 0; i<3; ++i)
		CV_MAT_ELEM(*b, float, i, 0) = CV_MAT_ELEM(*work_3x1, float, i, 0) + (float)c_color.val[i] * alpha*inv_sigmac_square;
	//
	cvMatMul(inv_bg_cov, bg_mean, work_3x1);
	for (int i = 0; i<3; ++i)
		CV_MAT_ELEM(*b, float, 3 + i, 0) = CV_MAT_ELEM(*work_3x1, float, i, 0) + (float)c_color.val[i] * (1 - alpha)*inv_sigmac_square;


	//
	cvSolve(A, b, x);

	CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 0, 0)));
	CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 1) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 1, 0)));
	CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 2) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 2, 0)));
	CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 3, 0)));
	CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 4, 0)));
	CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2) = (uchar)max(0, min(255, CV_MAT_ELEM(*x, float, 5, 0)));

	cvReleaseMat(&A), cvReleaseMat(&x), cvReleaseMat(&b), cvReleaseMat(&I), cvReleaseMat(&work_3x3), cvReleaseMat(&work_3x1);
}

inline void BayesianMatting::SolveAlpha(int r, int c)
{
	CV_IMAGE_ELEM(alphamap, float, r, c) =
		(((float)CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c))   * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c))
		+ ((float)CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c + 1) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1)) * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 1) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1))
		+ ((float)CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c + 2) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2)) * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 2) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2))
		) /
		(
		((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c))   * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c))
		+ ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 1) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1)) * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 1) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 1))
		+ ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 2) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2)) * ((float)CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + 2) - (float)CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + 2))
		);

	CV_IMAGE_ELEM(alphamap, float, r, c) = max(0, min(1, CV_IMAGE_ELEM(alphamap, float, r, c)));
}

float BayesianMatting::computeLikelihood(int r, int c, CvMat *fg_mean, CvMat *inv_fg_cov, CvMat *bg_mean, CvMat *inv_bg_cov)
{
	float fgL, bgL, cL;
	int i;
	float alpha = CV_IMAGE_ELEM(alphamap, float, r, c);

	CvMat *work3x1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *work1x3 = cvCreateMat(1, 3, CV_32FC1);
	CvMat *work1x1 = cvCreateMat(1, 1, CV_32FC1);
	CvMat *fg_color = cvCreateMat(3, 1, CV_32FC1);
	CvMat *bg_color = cvCreateMat(3, 1, CV_32FC1);
	CvMat *c_color = cvCreateMat(3, 1, CV_32FC1);
	for (i = 0; i<3; i++)
	{
		CV_MAT_ELEM(*fg_color, float, i, 0) = CV_IMAGE_ELEM(fgImg, uchar, r, 3 * c + i);
		CV_MAT_ELEM(*bg_color, float, i, 0) = CV_IMAGE_ELEM(bgImg, uchar, r, 3 * c + i);
		CV_MAT_ELEM(*c_color, float, i, 0) = CV_IMAGE_ELEM(colorImg, uchar, r, 3 * c + i);
	}

	// fgL
	cvSub(fg_color, fg_mean, work3x1);
	cvTranspose(work3x1, work1x3);
	cvMatMul(work1x3, inv_fg_cov, work1x3);
	cvMatMul(work1x3, work3x1, work1x1);
	fgL = -1.0f*CV_MAT_ELEM(*work1x1, float, 0, 0) / 2;

	// bgL
	cvSub(bg_color, bg_mean, work3x1);
	cvTranspose(work3x1, work1x3);
	cvMatMul(work1x3, inv_bg_cov, work1x3);
	cvMatMul(work1x3, work3x1, work1x1);
	bgL = -1.f*CV_MAT_ELEM(*work1x1, float, 0, 0) / 2;

	// cL
	cvAddWeighted(c_color, 1.0f, fg_color, -1.0f*alpha, 0.0f, work3x1);
	cvAddWeighted(work3x1, 1.0f, bg_color, -1.0f*(1.0f - alpha), 0.0f, work3x1);
	cL = -cvDotProduct(work3x1, work3x1) / (2 * sigmac * sigmac);

	cvReleaseMat(&work3x1);
	cvReleaseMat(&work1x3);
	cvReleaseMat(&work1x1);
	cvReleaseMat(&fg_color);
	cvReleaseMat(&bg_color);
	cvReleaseMat(&c_color);

	return cL + fgL + bgL;
}

float BayesianMatting::computeLikelihood(int r, int c, float fg_weight, CvMat *fg_mean, CvMat *inv_fg_cov, float bg_weight, CvMat *bg_mean, CvMat *inv_bg_cov)
{
	return computeLikelihood(r, c, fg_mean, inv_fg_cov, bg_mean, inv_bg_cov) + logf(fg_weight) + logf(bg_weight);
}



void mexFunction(const int nout, mxArray* output[], const int in, const mxArray* input[])
{
#ifndef _DEBUG
	omp_set_num_threads(std::max<int>(4, omp_get_max_threads()));
	omp_set_dynamic(std::max<int>(2, omp_get_max_threads() / 2));
#endif

	if (in != 3 || nout != 3 || mxGetClassID(input[0]) != mxUINT8_CLASS || mxGetClassID(input[1]) != mxUINT8_CLASS)
	{
		mexErrMsgTxt("USAGE: [Alpha, Fg, Bg] = simple_matting(uint8(Image), uint8(Trimap), N);");
	}

	MexImage<uint8_t> Image(input[0]);
	MexImage<uint8_t> Trimap(input[1]);
	//const float alpha = std::max<float>(0.f, std::min<float>(1.f, mxGetScalar(input[2])));
	const unsigned n = std::max<unsigned>(2u, std::min<unsigned>(100, mxGetScalar(input[2])));
	const int width = Image.width;
	const int height = Image.height;
	const long HW = Image.layer_size;
	const float nan = sqrt(-1.f);
	const size_t dims[] = { (size_t)height, (size_t)width, (size_t)colors };
	const size_t dims1[] = { (size_t)height, (size_t)width, 1 };
	output[0] = mxCreateNumericArray(3, dims1, mxSINGLE_CLASS, mxREAL);
	output[1] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	output[2] = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);
	MexImage<float> Alpha(output[0]);
	MexImage<uint8_t> Fg(output[1]);
	MexImage<uint8_t> Bg(output[2]);
	MexImage<float> DistFg(width, height);
	MexImage<float> DistBg(width, height);

	optimized_matting matting;

	DistFg.setval(nan);
	DistBg.setval(nan);
	//unsigned UNKNOWN = 0;
	//MexImage<unsigned> UMap(width, height);

#pragma omp parallel for
	for (long i = 0; i < HW; i++)
	{
		if (Trimap[i] == 0)
		{
			Alpha[i] = 0.f;
			for (int c = 0; c < colors; c++)
			{
				Bg[i + HW*c] = Image[i + HW*c];
			}
		}
		else if (Trimap[i] == 255 || Trimap[i] == 2)
		{
			Alpha[i] = 1.f;
			for (int c = 0; c < colors; c++)
			{
				Fg[i + HW*c] = Image[i + HW*c];
			}
		}
		else
		{
			const int x = i / height;
			const int y = i % height;
			matting.process(x, y, Image, Trimap, Alpha, Fg, Bg, n);
		}

	}


}