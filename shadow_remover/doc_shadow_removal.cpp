//#include"doc_shadow_removal.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#include <stdio.h>
#include<string>
#include <iostream>
using namespace cv;
using namespace std;


void ThresholdIntegral(Mat &inputMat, double thre, Mat &outputMat)
{
	// accept only char type matrices
	CV_Assert(!inputMat.empty());
	CV_Assert(inputMat.depth() == CV_8U);
	CV_Assert(inputMat.channels() == 1);
	CV_Assert(!outputMat.empty());
	CV_Assert(outputMat.depth() == CV_8U);
	CV_Assert(outputMat.channels() == 1);

	// rows -> height -> y
	int nRows = inputMat.rows;
	// cols -> width -> x
	int nCols = inputMat.cols;

	// create the integral image
	cv::Mat sumMat;
	cv::integral(inputMat, sumMat);

	CV_Assert(sumMat.depth() == CV_32S);
	CV_Assert(sizeof(int) == 4);

	int S = MAX(nRows, nCols) / 8;
	double T = 0.15;

	// perform thresholding
	int s2 = S / 2;
	int x1, y1, x2, y2, count, sum;

	// CV_Assert(sizeof(int) == 4);
	int *p_y1, *p_y2;
	uchar *p_inputMat, *p_outputMat;

	for (int i = 0; i < nRows; ++i)
	{
		y1 = i - s2;
		y2 = i + s2;

		if (y1 < 0) {
			y1 = 0;
		}
		if (y2 >= nRows) {
			y2 = nRows - 1;
		}

		p_y1 = sumMat.ptr<int>(y1);
		p_y2 = sumMat.ptr<int>(y2);
		p_inputMat = inputMat.ptr<uchar>(i);
		p_outputMat = outputMat.ptr<uchar>(i);

		for (int j = 0; j < nCols; ++j)
		{
			// set the SxS region
			x1 = j - s2;
			x2 = j + s2;

			if (x1 < 0) {
				x1 = 0;
			}
			if (x2 >= nCols) {
				x2 = nCols - 1;
			}

			count = (x2 - x1)*(y2 - y1);

			// I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
			sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

			if ((int)(p_inputMat[j] * count) < (int)(sum*(1.0 - T)*thre))
				p_outputMat[j] = 0;
			else
				p_outputMat[j] = 255;
		}
	}
}



void EvaluationIllumination(Mat& src, int size, Mat& dst, int L0Size)
{
	int height = src.rows;
	int width = src.cols;
//    int value = 0;
//	double fusionFactor = 0;
	int minH, minW, maxH, maxW;
	uchar* puSrcTemp;
    uchar* puSrc, *puDst;
    uchar r,g,b;
    int max_valueBGR[3], min_valueBGR[3];
    
    int maxWinSize = L0Size / sizeof(int) / 3;
    int maxHeightWin = ceil(height * 1.0 / maxWinSize);
    int maxWidthWin = ceil(width * 1.0 / maxWinSize);
    int iFrom, iTo, jFrom, jTo;
    for (int iWin = 0; iWin < maxHeightWin; iWin++)
    {
        iFrom = iWin * maxWinSize;
        iTo = min(iFrom + maxWinSize, height);
        for (int jWin = 0; jWin < maxWidthWin; jWin++)
        {
            jFrom = jWin * maxWinSize;
            jTo = min(jFrom + maxWinSize, width);
            
            
            for (int i = iFrom; i<iTo; i++)
            {
                puSrc = src.ptr(i);
                puDst = dst.ptr(i);
                minH = max(i - size, 0);
                maxH = min(i + size, height - 1);
                
                for (int j = jFrom; j<jTo; j++)
                {
                    minW = max(j - size, 0);
                    maxW = min(j + size, width - 1);
                    
                    memset(max_valueBGR, 0, sizeof(int) * 3);
                    memset(min_valueBGR, 0, sizeof(int) * 3);
                    
                    for (int ii = minH; ii <= maxH; ii++)
                    {
                        puSrcTemp = src.ptr(ii);
                        for (int jj = minW; jj <= maxW; jj++)
                        {
                            r = *(puSrcTemp + 3 * jj);
                            g = *(puSrcTemp + 3 * jj + 1);
                            b = *(puSrcTemp + 3 * jj + 2);
                            if (r > max_valueBGR[0])
                                max_valueBGR[0] = r;
                            if (g > max_valueBGR[1])
                                max_valueBGR[1] = g;
                            if (b > max_valueBGR[2])
                                max_valueBGR[2] = b;
                            
                            if (r < min_valueBGR[0])
                                min_valueBGR[0] = r;
                            if (g < min_valueBGR[1])
                                min_valueBGR[1] = g;
                            if (b < min_valueBGR[2])
                                min_valueBGR[2] = b;
                            /*
                            if (puSrcTemp[3 * jj] > max_valueBGR[0])
                                max_valueBGR[0] = puSrcTemp[3 * jj];
                            if (puSrcTemp[3 * jj + 1] > max_valueBGR[1])
                                max_valueBGR[1] = puSrcTemp[3 * jj + 1];
                            if (puSrcTemp[3 * jj + 2] > max_valueBGR[2])
                                max_valueBGR[2] = puSrcTemp[3 * jj + 2];
                            
                            if (puSrcTemp[3 * jj] < min_valueBGR[0])
                                min_valueBGR[0] = puSrcTemp[3 * jj];
                            if (puSrcTemp[3 * jj + 1] < min_valueBGR[1])
                                min_valueBGR[1] = puSrcTemp[3 * jj + 1];
                            if (puSrcTemp[3 * jj + 2] < min_valueBGR[2])
                                min_valueBGR[2] = puSrcTemp[3 * jj + 2];
                             */
                        }
                    }
                    
                    
                    for (int k = 0; k<3; k++)
                    {
                        if (max_valueBGR[k] > 0)
                        {
                            //					fusionFactor = (max_valueBGR[k] - min_valueBGR[k])*1.0 / max_valueBGR[k];
                            //					value = max_valueBGR[k] * fusionFactor + min_valueBGR[k] * (1 - fusionFactor);
                            //                  puDst[3 * j + k] = value /*(int)(255 * puSrc[3 * j + k] / value)*/;
                            *(puDst + 3 * j + k) = (max_valueBGR[k] - min_valueBGR[k]) + (min_valueBGR[k] * min_valueBGR[k]) / max_valueBGR[k];
                        }
                        else
                        {
                            *(puDst + 3 * j + k) = *(puSrc + 3 * j + k);
                        }
                        
                    }
                    
                }//j
                
            }//i
        }
    }

}

void EvaluationIlluminationV2(Mat& src, int size, Mat& dst)
{
    int height = src.rows;
    int width = src.cols;
    //    int value = 0;
    //    double fusionFactor = 0;
    int minH, minW, maxH, maxW;
    uchar* puSrcTemp;
    uchar* puSrc, *puDst;
    int max_valueBGR[3], min_valueBGR[3];
        
    for (int i = 0; i<height; i++)
    {
        puSrc = src.ptr(i);
        puDst = dst.ptr(i);
        minH = max(i - size, 0);
        maxH = min(i + size, height - 1);
        
        for (int j = 0; j<width; j++)
        {
            minW = max(j - size, 0);
            maxW = min(j + size, width - 1);
            
            memset(max_valueBGR, 0, sizeof(int) * 3);
            memset(min_valueBGR, 0, sizeof(int) * 3);
            
            for (int ii = minH; ii <= maxH; ii++)
            {
                puSrcTemp = src.ptr(ii);
                for (int jj = minW; jj <= maxW; jj++)
                {
                    if (puSrcTemp[3 * jj] > max_valueBGR[0])
                        max_valueBGR[0] = puSrcTemp[3 * jj];
                    if (puSrcTemp[3 * jj + 1] > max_valueBGR[1])
                        max_valueBGR[1] = puSrcTemp[3 * jj + 1];
                    if (puSrcTemp[3 * jj + 2] > max_valueBGR[2])
                        max_valueBGR[2] = puSrcTemp[3 * jj + 2];
                    
                    if (puSrcTemp[3 * jj] < min_valueBGR[0])
                        min_valueBGR[0] = puSrcTemp[3 * jj];
                    if (puSrcTemp[3 * jj + 1] < min_valueBGR[1])
                        min_valueBGR[1] = puSrcTemp[3 * jj + 1];
                    if (puSrcTemp[3 * jj + 2] < min_valueBGR[2])
                        min_valueBGR[2] = puSrcTemp[3 * jj + 2];
                }
            }
            
            
            for (int k = 0; k<3; k++)
            {
                if (max_valueBGR[k] > 0)
                {
                    //                    fusionFactor = (max_valueBGR[k] - min_valueBGR[k])*1.0 / max_valueBGR[k];
                    //                    value = max_valueBGR[k] * fusionFactor + min_valueBGR[k] * (1 - fusionFactor);
                    //                  puDst[3 * j + k] = value /*(int)(255 * puSrc[3 * j + k] / value)*/;
                    puDst[3 * j + k] = (max_valueBGR[k] - min_valueBGR[k]) + (min_valueBGR[k] * min_valueBGR[k]) / max_valueBGR[k];
                }
                else
                {
                    puDst[3 * j + k] = puSrc[3 * j + k];
                }
                
            }
            
        }//j
        
    }//i
}


double CalulateOneImgMSE(Mat& src,Mat& predict)
{
	double MSE = 0;
	int height = src.rows;
	int width = src.cols; 
	int iChannel = src.channels();
	int temp = 0;
    uchar* puSrc, *puPredict;

	if (iChannel == 3)
	{
		for (int i = 0; i<height; i++)
		{
			puSrc = src.ptr(i);
			puPredict = predict.ptr(i);
			for (int j = 0; j<width; j++)
			{
				for (int k = 0; k<3; k++)
				{
					temp = puSrc[3 * j + k] - puPredict[3 * j + k];
					MSE += temp*temp;
				}
			}
		}
		MSE = MSE / (height*width*3);
	}
	if (iChannel == 1)
	{
		for (int i = 0; i<height; i++)
		{
			puSrc = src.ptr(i);
			puPredict = predict.ptr(i);
			for (int j = 0; j<width; j++)
			{
				temp = puSrc[j] - puPredict[j];
				MSE += temp*temp;
			}
		}
		MSE = MSE / (height*width);
	}
	

	return MSE;
}

double GetMean(Mat& grayImg)
{
	int height = grayImg.rows;
	int width = grayImg.cols;
	double sum = 0;
	double mean = 0;
    uchar* puGrayImg;
	for (int i = 0; i < height; i++)
	{
		puGrayImg = grayImg.ptr(i);
		for (int j = 0; j < width; j++)
		{
			sum += puGrayImg[j];
		}
	}
	mean = sum / (height*width);

	return mean;
}


void FindReferenceBg(Mat& bgImg, Mat& binary,Mat& shadowMap, int iRefBg[3])
 {
	int height = bgImg.rows;
	int width = bgImg.cols;
	double BGR[3] = { 0 };
	double countNum = 0;
    uchar* puBgImg, *puBinary, *puShadowMap;

	for (int i = 0; i < height; i++)
	{
		puBgImg = bgImg.ptr(i);
		puBinary = binary.ptr(i);
		puShadowMap = shadowMap.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j]>0 && puBinary[j]>0)
			{
				BGR[0] += puBgImg[3 * j];
				BGR[1] += puBgImg[3 * j+1];
				BGR[2] += puBgImg[3 * j+2];
				countNum++;
			}
		}
	}

	int avg_bgr[3];
	for (int i=0;i<3;i++)
	{
		avg_bgr[i] = BGR[i] / countNum; 
	}
	
	double curMin = 255 * 255 * 3;
	double diff=0,curMag=0;
	for (int i = 0; i < height; i++)
	{
		puBgImg = bgImg.ptr(i);
		puShadowMap = shadowMap.ptr(i);
		puBinary = binary.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j]>0 && puBinary[j]>0)
			{
				curMag = 0;
				for (int k = 0; k < 3; k++) 
				{
					diff = puBgImg[k] - avg_bgr[k];
					curMag += diff * diff;
				}
				if (curMag < curMin)
				{
					curMin = curMag;
					iRefBg[0] = puBgImg[3*j];
					iRefBg[1] = puBgImg[3*j+1];
					iRefBg[2] = puBgImg[3*j+2];
				}

			}
		}
	}

}


/*******************************************
*Function: Remove the shadow by bg color ratio
*Input: img, RGB channels, original image
*       localBgColorImg, RGB channels, local background image
*       iRefBg, reference global bg
*Output: result, RGB channels, image without shadows
*return void
*date: 2019.01.15   wangbingshu
********************************************/
void RemovalShadowByBgColorRatio(Mat& img, Mat& localBgColorImg, int iRefBg[3], Mat& result)
{
	int height = img.rows;
	int width = img.cols;
	double ratio;
    uchar* puImg, *puLocalRef, *puResult;

	for (int i = 0; i < height; i++)
	{
		puImg = img.ptr(i);
		puLocalRef = localBgColorImg.ptr(i);
		puResult= result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			for (int k=0;k<3;k++)
			{
				ratio = 1.0*puLocalRef[3 * j+k] / iRefBg[k];
				puResult[3 * j+k] = puImg[3 * j+k]/ratio;
			}
		}
	}
}



void CalculateShadowStrengthFactor(Mat& localBgColorImg, Mat& binaryImg, Mat& shadowMap, double dSSF[3])
{
	dSSF[0] = 1;
	dSSF[1] = 1;
	dSSF[2] = 1;

	Mat erodeBinary(shadowMap.size(), CV_8UC1, 1);
	Mat dilateBinary(shadowMap.size(), CV_8UC1, 1);
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	erode(shadowMap, erodeBinary, element);
	dilate(shadowMap, dilateBinary, element);

	int height = localBgColorImg.rows;
	int width = localBgColorImg.cols;
	double bright[3] = { 0 };
	double brightNum = 0;

	double dark[3] = { 0 };
	double darkNum = 0;
    uchar* puBgColorImg, *puErodeBinary, *puDilateBinary, *puBinary;

	for (int i = 0; i < height; i++)
	{
		puBgColorImg = localBgColorImg.ptr(i);
		puErodeBinary = erodeBinary.ptr(i);
		puDilateBinary = dilateBinary.ptr(i);
		puBinary = binaryImg.ptr(i);

		for (int j = 0; j < width; j++)
		{
			if (puErodeBinary[j]>0 && puBinary[j]>0)
			{
				bright[0] += puBgColorImg[3 * j];
				bright[1] += puBgColorImg[3 * j + 1];
				bright[2] += puBgColorImg[3 * j + 2];
				brightNum++;
			}
			if (puDilateBinary[j] == 0 && puBinary[j]>0)
			{
				dark[0] += puBgColorImg[3 * j];
				dark[1] += puBgColorImg[3 * j + 1];
				dark[2] += puBgColorImg[3 * j + 2];
				darkNum++;
			}
		}
	}
	if (darkNum>0&&brightNum>0)
	{
		for (int i=0;i<3;i++)
		{
			bright[i] = 1.0* bright[i] / brightNum;
			dark[i] = 1.0*dark[i] / darkNum;
			dSSF[i] = bright[i] / dark[i];
		}
	}
}


void FillHole(Mat &src, Mat& shadowMap, double dSSF[3], int iRefBg[3], Mat &result)
{
	Mat gray(src.size(), CV_8UC1, 1);
	Mat binary(src.size(), CV_8UC1, 1);
	cvtColor(src, gray, COLOR_BGR2GRAY);
	ThresholdIntegral(gray, 1.0, binary);

	Mat erodeBinary(src.size(), CV_8UC1, 1);
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	gray.copyTo(erodeBinary);
	erode(binary, erodeBinary, element);

	double ratio[3] = { 1 };
	double countNum[3] = { 0 };
	int height = src.rows;
	int width = src.cols;
    uchar* puSrc, *puShadowMap, *puErodeBinary, *puResult;

	for (int i = 0; i < height; i++)
	{
		puSrc = src.ptr(i);
		puShadowMap = shadowMap.ptr(i);
		puErodeBinary = erodeBinary.ptr(i);
		puResult = result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			if (puShadowMap[j] == 255 && puErodeBinary[j] == 255)
			{
				for (int k=0;k<3;k++)
				{
					if (puSrc[3 * j + k] > 0)
					{
						ratio[k] += 1.0*puResult[3 * j + k] / puSrc[3 * j + k];
						countNum[k]++;
					}
				}
			}
		}
	}

	for (int i=0;i<3;i++)
	{
		if (countNum[i]>0)
		{
			ratio[i] = ratio[i] / countNum[i];
		}
		if (dSSF[i] > 1.3)
		{
			dSSF[i] = (dSSF[i] + 1) / 2;
		}
	}
	 
	if (ratio[0]>0 && ratio[1]>0 && ratio[2]>0&& countNum[0]>0 &&countNum[1]>0 && countNum[2]>0)
	{
        uchar *puBinary;
		for (int i = 0; i < height; i++)
		{
			puResult = result.ptr(i);
			puSrc = src.ptr(i);
			puBinary = binary.ptr(i);
			puShadowMap = shadowMap.ptr(i);
		 
			for (int j = 0; j < width; j++)
			{
				if (puShadowMap[j]==0 && puBinary[j] == 0)
				{
				   for (int k=0;k<3;k++)
					{
						puResult[3 * j + k] =  dSSF[k] * puSrc[3 * j + k] / ratio[k];
					}
				}
				if (puShadowMap[j] == 255 && puBinary[j] == 0)
				{
					puResult[3 * j] = puSrc[3 * j] ;
					puResult[3 * j + 1] = puSrc[3 * j + 1];
					puResult[3 * j + 2] = puSrc[3 * j + 2];
				}
			}
		}
	}

}

void ToneAdjust(Mat& img, Mat& binaryImg,Mat& shadowMap, double dSSF[3], Mat& result)
{
	//cout << dSSF[0] << " " << dSSF[1] << " " << dSSF[2] << endl;
	int height = img.rows;
	int width = img.cols;
	int size = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * size + 1, 2 * size + 1), Point(size, size));
	if ((dSSF[0]+ dSSF[0]+ dSSF[0])/3<2)
	{
		erode(shadowMap, shadowMap, element);
	}

	
	double ratioBGR[3] = {0};
	int diffBGR[3] = {0};
	double ratioTemp;
    uchar* puImg, *puBinaryImg, *puShadowMap, *puResult;

	for (int i = 0; i < height; i++)
	{
		puImg = img.ptr(i);
		puBinaryImg = binaryImg.ptr(i);
		puShadowMap = shadowMap.ptr(i);
		puResult = result.ptr(i);
		for (int j = 0; j < width; j++)
		{
			
			if (puBinaryImg[j] == 0 && puShadowMap[j]==0)
			{
				for (int k=0;k<3;k++)
				{
					diffBGR[k] = puResult[3 * j + k] - puImg[3 * j + k];
					ratioBGR[k] = 1.0*(puResult[3 * j + k] + 1) / (puImg[3 * j + k] + 1);
					
					if (puImg[3 * j + k]<8)
					{
						ratioTemp = dSSF[k]+1;
						puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					}
					else if (puImg[3 * j + k]<15)
					{
						ratioTemp = dSSF[k];
						puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					}else
				    if(puImg[3 * j + k]<30 )
					 {
						 ratioTemp = (1 + ratioBGR[k]) / 2;
						 puResult[3 * j + k] = puImg[3 * j + k] * ratioTemp;
					 }
					else if(puImg[3 * j + k]<60 && ratioBGR[k]>dSSF[k])
					{
						puResult[3 * j + k] = puImg[3 * j + k] * (1 + dSSF[k]) / 2;
					}else
					{
						 ;
					}
					
				}
			}

			
			if (puBinaryImg[j] == 0 && puShadowMap[j] == 255)
			{
				for (int k = 0; k < 3; k++)
				{
					puResult[3 * j + k] = puImg[3 * j + k];
				}
			}


		}
	}

}

void printDiffTime(clock_t start, string title)
{
    printf("%.0f ms <-- %s\n", (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000), title.c_str());
}

//#define VERBOSE
//#define USEL0
double executeShadowRemovalAction(Mat& img, Mat& result, int L0Size)
{
	Mat img2(img.size(), CV_8UC3, 3);
	img.copyTo(img2);
    clock_t globalStart;
#if defined(VERBOSE)
    clock_t start;
#endif
    globalStart = clock();
    
	Mat gray(img.size(), CV_8UC1, 1);
	Mat binaryImg(img.size(), CV_8UC1, 1);
	cvtColor(img, gray, COLOR_BGR2GRAY);
#if defined(VERBOSE)
    start = clock();
#endif
	ThresholdIntegral(gray,1.0, binaryImg);  //
#if defined(VERBOSE)
    printDiffTime(start, "ThresholdIntegral");
#endif

	//imshow("binaryImg", binaryImg);
	//waitKey(0);

    // Get the local bg image and the shadow map
	Mat bg(Size(img.cols, img.rows), CV_8UC3, 3);
	Mat shadowMap(img.size(), CV_8UC1, 1);
	for (int i = 0; i < 3; i++)
	{
//        cv::imshow("EvaluationIllumination - " + to_string(i), img);
//        waitKey();
#if defined(VERBOSE)
        start = clock();
#endif
        
#if defined(USEL0)
        EvaluationIllumination(img, 2, bg, L0Size);  //				  //imshow("imgResult", imgResult);
#else
        EvaluationIlluminationV2(img, 2, bg);  //                  //imshow("imgResult", imgResult);
#endif
		bg.copyTo(img);
#if defined(VERBOSE)
        printDiffTime(start, "EvaluationIllumination - " + to_string(i + 1));
#endif
	}
//    cv::imshow("Final EvaluationIllumination", img);
//    waitKey();
#if defined(VERBOSE)
    start = clock();
#endif
	cvtColor(bg, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 3); 
	threshold(gray, shadowMap, 0, 255, THRESH_OTSU);   
#if defined(VERBOSE)
    printDiffTime(start, "Get the local bg image and the shadow map");
#endif

    
	int iRefBg[3] = { 0 };
#if defined(VERBOSE)
    start = clock();
#endif
	FindReferenceBg(bg, binaryImg,shadowMap, iRefBg);
#if defined(VERBOSE)
    printDiffTime(start, "FindReferenceBg");
#endif

#if defined(VERBOSE)
    start = clock();
#endif
	Mat localBgColorImg(Size(img.cols, img.rows), CV_8UC3, 3);
#if defined(USEL0)
    EvaluationIllumination(img2, 1, localBgColorImg, L0Size);  //                  //imshow("imgResult", imgResult);
#else
    EvaluationIlluminationV2(img2, 1, localBgColorImg);  //                  //imshow("imgResult", imgResult);
#endif
#if defined(VERBOSE)
    printDiffTime(start, "EvaluationIllumination");
#endif

#if defined(VERBOSE)
    start = clock();
#endif
	RemovalShadowByBgColorRatio(img2, localBgColorImg, iRefBg, result);
#if defined(VERBOSE)
    printDiffTime(start, "RemovalShadowByBgColorRatio");
#endif
	//imshow("result", result);
	
	double dSSF[3] = { 0 };
#if defined(VERBOSE)
    start = clock();
#endif
	CalculateShadowStrengthFactor(localBgColorImg, binaryImg,shadowMap, dSSF);
#if defined(VERBOSE)
    printDiffTime(start, "CalculateShadowStrengthFactor");
#endif
	//cout << dSSF[0] << " " << dSSF[1] << " " << dSSF[2] << endl;
	double avg = (dSSF[0] + dSSF[1] + dSSF[2]) / 3;
	if (avg<1.39 )
	{
#if defined(VERBOSE)
        start = clock();
#endif
		FillHole(img2,  shadowMap, dSSF, iRefBg, result);
#if defined(VERBOSE)
        printDiffTime(start, "FillHole");
#endif
	}

	
#if defined(VERBOSE)
    start = clock();
#endif
	ToneAdjust(img2,binaryImg, shadowMap,dSSF,result);
#if defined(VERBOSE)
    printDiffTime(start, "ToneAdjust");
#endif
	img2.copyTo(img);

    //printDiffTime(globalStart, "+++ShadowRemoval total time");
    return (std::clock() - globalStart) / (double)(CLOCKS_PER_SEC / 1000);
//    cv::imshow("after shadow removal", result);
//    waitKey();
}

#define KB (int) 1024
#define MB (int) 1024 * 1024
#define GB 1024 * 1024 * 1024
void ShadowRemoval(Mat& img, Mat& result)
{
    int sizes[] = {
            1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB, 32 * KB, 64 * KB, 128 * KB,
            192 * KB, 256 * KB, 320 * KB, 380 * KB, 448 * KB, 512 * KB,
            1 * MB, 2 * MB, 4 * MB, 6 * MB, 8 * MB, 10 * MB, 12 * MB, 14 * MB,
            16 * MB, 18 * MB, 20 * MB, 22 * MB, 24 * MB, 26 * MB, 28 * MB, 30 * MB,
            32 * MB, 64 * MB
        };
    double elapsedTime = 0;
    for ( long i = 0; i < 32; i ++) {
        elapsedTime = executeShadowRemovalAction(img, result, sizes[i]);
        printf("%.0f, %d\n", elapsedTime, sizes[i] / KB);
    }
}
