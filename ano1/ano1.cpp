// 
//

#include "stdafx.h"

void cv1() // detekce hrany
{
	cv::Mat src_8uc1_img;
	cv::Mat fx;
	cv::Mat fy;
	cv::Mat exy;
	cv::Mat sobel;
	float Gx, Gy;
	int GxMat[] = {-1,0,1,-2,0,2,-1,0,1};
	int GyMat[] = {-1,-2,-1,0,0,0,1,2,1};

	src_8uc1_img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	src_8uc1_img.convertTo(src_8uc1_img, CV_32FC1, 1 / 255.0);

	fx = src_8uc1_img.clone();
	fy = src_8uc1_img.clone();
	exy = src_8uc1_img.clone();
	sobel = src_8uc1_img.clone();


	// vypocet fx,fy,exy
	for (int x = 1; x < src_8uc1_img.rows; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols; y++)
		{
			fx.at<float>(x, y) = src_8uc1_img.at<float>(x, y) - src_8uc1_img.at<float>(x - 1, y);
			fy.at<float>(x, y) = src_8uc1_img.at<float>(x, y) - src_8uc1_img.at<float>(x, y - 1);
			exy.at<float>(x, y) = sqrt(fx.at<float>(x, y) * fx.at<float>(x, y) + fy.at<float>(x, y) * fy.at<float>(x, y));
		}
	}

	//vypocet sobela
	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			Gx = src_8uc1_img.at<float>(x - 1, y - 1) * GxMat[0] + src_8uc1_img.at<float>(x - 1, y) * GxMat[1] + src_8uc1_img.at<float>(x - 1, y + 1) * GxMat[2] +
				src_8uc1_img.at<float>(x, y - 1) * GxMat[3] + src_8uc1_img.at<float>(x, y) * GxMat[4] + src_8uc1_img.at<float>(x, y + 1) * GxMat[5] +
				src_8uc1_img.at<float>(x + 1, y - 1) * GxMat[6] + src_8uc1_img.at<float>(x + 1, y) * GxMat[7] + src_8uc1_img.at<float>(x + 1, y + 1) * GxMat[8];

			Gy = src_8uc1_img.at<float>(x - 1, y - 1) * GyMat[0] + src_8uc1_img.at<float>(x - 1, y) * GyMat[1] + src_8uc1_img.at<float>(x - 1, y + 1) * GyMat[2] +
				src_8uc1_img.at<float>(x, y - 1) * GyMat[3] + src_8uc1_img.at<float>(x, y) * GyMat[4] + src_8uc1_img.at<float>(x, y + 1) * GyMat[5] +
				src_8uc1_img.at<float>(x + 1, y - 1) * GyMat[6] + src_8uc1_img.at<float>(x + 1, y) * GyMat[7] + src_8uc1_img.at<float>(x + 1, y + 1) * GyMat[8];

			sobel.at<float>(x, y) = sqrt(Gx * Gx + Gy * Gy);
		}
	}

	cv::imshow("SRC", src_8uc1_img); // display image
	cv::imshow("fx", fx); // display image
	cv::imshow("fy", fy); // display image
	cv::imshow("exy", exy); // display image
	cv::imshow("sobel", sobel); // display image
}

void cv2()
{
	cv::Mat src_8uc1_img;
	cv::Mat fx;
	cv::Mat fy;
	cv::Mat exy;
	cv::Mat exyc;
	cv::Mat gaus;
	cv::Mat gausLap;
	float Gy;
	int GxMat[] = {0, 1, 0, 1, -4, 1, 0, 1, 0};
	int GausMat[] = {0, 1, 0, 1, 4, 1, 0, 1, 0};
	int GausMat2[] = {0, 0, 1, 0, 0,
		0,1,2,1,0,
		1,2,-16,2,1,
		0,1,2,1,0,
		0,0,1,0,0};

	src_8uc1_img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	src_8uc1_img.convertTo(src_8uc1_img, CV_32FC1, 1 / 255.0);

	fx = src_8uc1_img.clone();
	fy = src_8uc1_img.clone();
	exy = src_8uc1_img.clone();
	exyc = src_8uc1_img.clone();
	gaus = src_8uc1_img.clone();
	gausLap = src_8uc1_img.clone();


	// vypocet fx,fy,exy
	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			fx.at<float>(x, y) = src_8uc1_img.at<float>(x - 1, y) - 2 * src_8uc1_img.at<float>(x, y) + src_8uc1_img.at<float>(x + 1, y);
			fy.at<float>(x, y) = src_8uc1_img.at<float>(x, y - 1) - 2 * src_8uc1_img.at<float>(x, y) + src_8uc1_img.at<float>(x, y + 1);
			//exy.at<float>(x, y) = sqrt(fx.at<float>(x, y)*fx.at<float>(x, y) + fy.at<float>(x, y)*fy.at<float>(x, y));
			exy.at<float>(x, y) = fx.at<float>(x, y) + fy.at<float>(x, y);
		}
	}

	//vypocet gausiana
	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			gaus.at<float>(x, y) = (src_8uc1_img.at<float>(x - 1, y - 1) * GxMat[0] + src_8uc1_img.at<float>(x - 1, y) * GxMat[1] + src_8uc1_img.at<float>(x - 1, y + 1) * GxMat[2] +
				src_8uc1_img.at<float>(x, y - 1) * GxMat[3] + src_8uc1_img.at<float>(x, y) * GxMat[4] + src_8uc1_img.at<float>(x, y + 1) * GxMat[5] +
				src_8uc1_img.at<float>(x + 1, y - 1) * GxMat[6] + src_8uc1_img.at<float>(x + 1, y) * GxMat[7] + src_8uc1_img.at<float>(x + 1, y + 1) * GxMat[8]);
		}
	}

	float fxA, fyA;
	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			exyc.at<float>(x, y) = (gaus.at<float>(x - 1, y - 1) * GausMat[0] + gaus.at<float>(x - 1, y) * GausMat[1] + gaus.at<float>(x - 1, y + 1) * GausMat[2] +
				gaus.at<float>(x, y - 1) * GausMat[3] + gaus.at<float>(x, y) * GausMat[4] + gaus.at<float>(x, y + 1) * GausMat[5] +
				gaus.at<float>(x + 1, y - 1) * GausMat[6] + gaus.at<float>(x + 1, y) * GausMat[7] + gaus.at<float>(x + 1, y + 1) * GausMat[8]) / 8;
		}
	}

	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			exyc.at<float>(x, y) = (gaus.at<float>(x - 1, y - 1) * GausMat[0] + gaus.at<float>(x - 1, y) * GausMat[1] + gaus.at<float>(x - 1, y + 1) * GausMat[2] +
				gaus.at<float>(x, y - 1) * GausMat[3] + gaus.at<float>(x, y) * GausMat[4] + gaus.at<float>(x, y + 1) * GausMat[5] +
				gaus.at<float>(x + 1, y - 1) * GausMat[6] + gaus.at<float>(x + 1, y) * GausMat[7] + gaus.at<float>(x + 1, y + 1) * GausMat[8]) / 8;
		}
	}

	for (int x = 2; x < src_8uc1_img.rows - 2; x++)
	{
		for (int y = 2; y < src_8uc1_img.cols - 2; y++)
		{
			gausLap.at<float>(x, y) = (src_8uc1_img.at<float>(x - 2, y - 2) * GausMat2[0] + src_8uc1_img.at<float>(x - 2, y - 1) * GausMat2[1] + src_8uc1_img.at<float>(x - 2, y) * GausMat2[2] + src_8uc1_img.at<float>(x - 2, y + 1) * GausMat2[3] + src_8uc1_img.at<float>(x - 2, y + 2) * GausMat2[4] +
				src_8uc1_img.at<float>(x - 1, y - 2) * GausMat2[5] + src_8uc1_img.at<float>(x - 1, y - 1) * GausMat2[6] + src_8uc1_img.at<float>(x - 1, y) * GausMat2[7] + src_8uc1_img.at<float>(x - 1, y + 1) * GausMat2[8] + src_8uc1_img.at<float>(x - 1, y + 2) * GausMat2[9] +
				src_8uc1_img.at<float>(x, y - 2) * GausMat2[10] + src_8uc1_img.at<float>(x, y - 1) * GausMat2[11] + src_8uc1_img.at<float>(x, y) * GausMat2[12] + src_8uc1_img.at<float>(x, y + 1) * GausMat2[13] + src_8uc1_img.at<float>(x, y + 2) * GausMat2[14] +
				src_8uc1_img.at<float>(x + 1, y - 2) * GausMat2[15] + src_8uc1_img.at<float>(x + 1, y - 1) * GausMat2[16] + src_8uc1_img.at<float>(x + 1, y) * GausMat2[17] + src_8uc1_img.at<float>(x + 1, y + 1) * GausMat2[18] + src_8uc1_img.at<float>(x + 1, y + 2) * GausMat2[19] +
				src_8uc1_img.at<float>(x + 2, y - 2) * GausMat2[20] + src_8uc1_img.at<float>(x + 2, y - 1) * GausMat2[21] + src_8uc1_img.at<float>(x + 2, y) * GausMat2[22] + src_8uc1_img.at<float>(x + 2, y + 1) * GausMat2[23] + src_8uc1_img.at<float>(x + 2, y + 2) * GausMat2[24]) / 8;
		}
	}

	cv::imshow("SRC", src_8uc1_img); // display image
	cv::imshow("fx", fx); // display image
	cv::imshow("fy", fy); // display image
	cv::imshow("exy", exy); // display image
	cv::imshow("gaus", gaus); // display image
	cv::imshow("gaus,laplace", exyc); // display image
	cv::imshow("gausLap", gausLap); // display image
}


const int slider_max = 100;
int t1_slider, t2_slider;
double alpha;
double beta;
cv::Mat epmFinal;
cv::Mat exy;
cv::Mat epmFinalT;

bool pointOutside(cv::Mat mattrix, int x, int y)
{
	if (mattrix.at<float>(x - 1, y - 1) == 1) return true;
	if (mattrix.at<float>(x, y - 1) == 1) return true;
	if (mattrix.at<float>(x + 1, y - 1) == 1) return true;
	if (mattrix.at<float>(x - 1, y) == 1) return true;
	if (mattrix.at<float>(x + 1, y) == 1) return true;
	if (mattrix.at<float>(x - 1, y + 1) == 1) return true;
	if (mattrix.at<float>(x, y + 1) == 1) return true;
	if (mattrix.at<float>(x + 1, y + 1) == 1) return true;
	return false;
}

void on_Change(int, void*)
{
	//float t1 = (float)t1_slider/100.0f;
	float t1 = 1.f / (float)t1_slider;
	//float t2 =(float)t2_slider/100.0f;
	float t2 = 1.f / (float)t2_slider;

	//

	for (int x = 1; x < epmFinal.rows; x++)
	{
		for (int y = 1; y < epmFinal.cols; y++)
		{
			if (exy.at<float>(x, y) > t1 && epmFinal.at<float>(x, y) == 1)
			{
				epmFinalT.at<float>(x, y) = 1;
			}
			else
			{
				epmFinalT.at<float>(x, y) = 0;
			}
		}
	}
	for (int x = 1; x < epmFinal.rows-1; x++)
	{
		for (int y = 1; y < epmFinal.cols-1; y++)
		{
			if (exy.at<float>(x, y) > t2 && pointOutside(epmFinalT, x, y) && epmFinal.at<float>(x, y) == 1)
			{
				//std::cout << x << y << std::endl;
				epmFinalT.at<float>(x, y) = 1;
			}
				
			
		}
	}

	for (int x = epmFinal.rows - 2; x > 0; x--)
	{
		for (int y = epmFinal.cols - 2; y >  0; y--)
		{
			if (exy.at<float>(x, y) > t2 && pointOutside(epmFinalT, x, y) && epmFinal.at<float>(x, y) == 1)
			{
				//std::cout << x << y << std::endl;
				epmFinalT.at<float>(x, y) = 1;
			}


		}
	}


	cv::imshow("Tresholds", epmFinalT);
}


void cv3()
{
	cv::Mat src_8uc1_img;
	src_8uc1_img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	src_8uc1_img.convertTo(src_8uc1_img, CV_32FC1, 1 / 255.0);

	cv::Mat arcTangMath;
	exy = src_8uc1_img.clone();
	epmFinal = src_8uc1_img.clone();
	arcTangMath = src_8uc1_img.clone();
	epmFinalT = src_8uc1_img.clone();


	float fx, fy;


	for (int x = 1; x < src_8uc1_img.rows; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols; y++)
		{
			fx = src_8uc1_img.at<float>(x, y) - src_8uc1_img.at<float>(x - 1, y);
			fy = src_8uc1_img.at<float>(x, y) - src_8uc1_img.at<float>(x, y - 1);
			arcTangMath.at<float>(x, y) = atan2(fy, fx);
			exy.at<float>(x, y) = sqrt(fx * fx + fy * fy);
		}
	}

	float ePlus, eMinus;
	float alfa;

	for (int x = 1; x < src_8uc1_img.rows - 1; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols - 1; y++)
		{
			float theta = arcTangMath.at<float>(x, y);

			ePlus = eMinus = 99999.0f;

			if ((theta < M_PI_4 && theta >= 0) || (theta >= -M_PI && theta < (-M_PI_2 -M_PI_4)))
			{
				alfa = tan(arcTangMath.at<float>(x, y));
				ePlus = alfa * exy.at<float>(x + 1, y + 1) + (1 - alfa) * exy.at<float>(x + 1, y);
				eMinus = alfa * exy.at<float>(x - 1, y - 1) + (1 - alfa) * exy.at<float>(x - 1, y);
			}
			else if ((theta < M_PI_2 && theta >= M_PI_4) || (theta >= -(M_PI_2 + M_PI_4) && theta < -(M_PI_2)))
			{
				alfa = tan(arcTangMath.at<float>(x, y) - M_PI_4);
				ePlus = alfa * exy.at<float>(x, y + 1) + (1 - alfa) * exy.at<float>(x + 1, y + 1);
				eMinus = alfa * exy.at<float>(x, y - 1) + (1 - alfa) * exy.at<float>(x - 1, y - 1);
			}
			else if ((theta < (M_PI_2 + M_PI_4) && theta >= M_PI_2) || (theta >= -(M_PI_2) && theta < -(M_PI_4)))
			{
				alfa = tan(arcTangMath.at<float>(x, y) - M_PI_2);
				ePlus = alfa * exy.at<float>(x - 1, y + 1) + (1 - alfa) * exy.at<float>(x, y + 1);
				eMinus = alfa * exy.at<float>(x + 1, y - 1) + (1 - alfa) * exy.at<float>(x, y - 1);
			}
			else if ((theta <= M_PI && theta >= (M_PI_2 + M_PI_4)) || (theta >= -(M_PI_4) && theta < 0))
			{
				alfa = tan(arcTangMath.at<float>(x, y) - (M_PI_2 + M_PI_4));
				ePlus = alfa * exy.at<float>(x - 1, y) + (1 - alfa) * exy.at<float>(x - 1, y + 1);
				eMinus = alfa * exy.at<float>(x + 1, y) + (1 - alfa) * exy.at<float>(x + 1, y - 1);
			}


			//std::cout <<"Exy: "<< exy.at<float>(x, y) << "  ePlus: " << ePlus << "  eMinus: " << eMinus << std::endl;
			if (exy.at<float>(x, y) > ePlus && exy.at<float>(x, y) > eMinus)
			{
				epmFinal.at<float>(x, y) = 1;
			}
			else
			{
				epmFinal.at<float>(x, y) = 0;
			}
		}
	}

	/// Initialize values
	t1_slider = 6;
	t2_slider = 10;

	/// Create Windows
	cv::namedWindow("Tresholds", 1);

	cv::createTrackbar("t1", "Tresholds", &t1_slider, slider_max, on_Change);
	cv::createTrackbar("t2", "Tresholds", &t2_slider, slider_max, on_Change);
	//cv::createTrackbar("red low", "Tresholds", NULL, 255, fooCallback, new ColorThresholdType(RED_LOW));


	/// Show some stuff
	on_Change(t1_slider, 0);

	for (int x = 1; x < src_8uc1_img.rows; x++)
	{
		for (int y = 1; y < src_8uc1_img.cols; y++)
		{
		}
	}

	//cv::imshow("SRC", src_8uc1_img); // display image
	cv::imshow("final", epmFinal); // display image
	//cv::imshow("exy", exy);
}

bool indexation(cv::Mat mat, cv::Mat indexace, int row, int col, int index)
{ 
	bool raiseIndex = false;
	if (!(row >= 0 && row < mat.rows && col >= 0 && col < mat.cols)) return false;
	if (mat.at<uchar>(row, col) == 0 || indexace.at<uchar>(row,col) != 0) return false;
	raiseIndex = true;

//	std::cout << row << "  " << col << "   " << index << std::endl;
	indexace.at<uchar>(row, col) = index;
	raiseIndex += indexation(mat, indexace, row + 1, col + 1, index);
	raiseIndex += indexation(mat, indexace, row - 1, col - 1, index);
	raiseIndex += indexation(mat, indexace, row + 1, col - 1, index);
	raiseIndex += indexation(mat, indexace, row - 1, col + 1, index);
	raiseIndex += indexation(mat, indexace, row, col + 1, index);
	raiseIndex += indexation(mat, indexace, row - 1, col, index);
	raiseIndex += indexation(mat, indexace, row + 1, col, index);
	raiseIndex += indexation(mat, indexace, row, col - 1, index);

	
	
	return raiseIndex;
}

void cv4()
{
	cv::Mat src_8uc1_img;
	cv::Mat img1;
	cv::Mat indexace;
	uchar tr = 100;
	
	src_8uc1_img = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	
	img1 = src_8uc1_img.clone();
	indexace = src_8uc1_img.clone();

	for (int x = 0; x < src_8uc1_img.rows; x++)
	{
		for (int y = 0; y < src_8uc1_img.cols; y++)
		{
			indexace.at<uchar>(x, y) = 0;
			if (src_8uc1_img.at<uchar>(x, y)>tr)
			{
				img1.at<uchar>(x, y) = 255;
			}
			else
			{
				img1.at<uchar>(x, y) = 0;
			}

		}
	}

	int index = 1;
	for (int x = 0; x < img1.rows; x++)
	{
		for (int y = 0; y < img1.cols; y++)
		{
			if (indexation(img1, indexace, x, y, index))
			{
				index++;
			}
		}
	}
	
    cv::imshow("SRC", src_8uc1_img); // display image
    cv::imshow("img1", img1); // display image
	cv::imshow("indexation", indexace);
	
}

int m(cv::Mat indexace, int p, int q, int index)
{
	int m = 0;
	
	for (int y = 0; y < indexace.rows; y++)
	{
		for (int x = 0; x < indexace.cols; x++)
		{
		     if(indexace.at<uchar>(y, x) == index)
			{
				m += pow(x,p)*pow(y,q);
			}
		
		}
	}
	return m;
}

float u(cv::Mat indexace, int p, int q,float xt, float yt, int index)
{
	float m = 0;

	for (int y = 0; y < indexace.rows; y++)
	{
		for (int x = 0; x < indexace.cols; x++)
		{
			if (indexace.at<uchar>(y, x) == index)
			{
				m += (pow(x - xt, p)*pow(y - yt, q));
			}

		}
	}
	return m;
}

void cv5()
{
	cv::Mat src_8uc1_img;
	cv::Mat img1;
	cv::Mat indexace;
	uchar tr = 100;

	src_8uc1_img = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale

	img1 = src_8uc1_img.clone();
	indexace = src_8uc1_img.clone();

	for (int x = 0; x < src_8uc1_img.rows; x++)
	{
		for (int y = 0; y < src_8uc1_img.cols; y++)
		{
			indexace.at<uchar>(x, y) = 0;
			if (src_8uc1_img.at<uchar>(x, y)>tr)
			{
				img1.at<uchar>(x, y) = 255;
			}
			else
			{
				img1.at<uchar>(x, y) = 0;
			}

		}
	}

	int index = 1;
	for (int x = 0; x < img1.rows; x++)
	{
		for (int y = 0; y < img1.cols; y++)
		{
			if (indexation(img1, indexace, x, y, index))
			{
				index++;
			}
		}
	}

	float xt = 0, yt = 0;
	float uMin = 0, uMax = 0;
	for (int i = 1; i < index; i++)
	{
		xt = (float)m(indexace, 1, 0, i) / m(indexace, 0, 0, i);
		yt = (float)m(indexace, 0, 1, i) / m(indexace, 0, 0, i);
		//std::cout << "xt,yt: "<< xt << "   " << yt << std::endl;


		uMax = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) + 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)),2));
		uMin = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) - 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)),2));

		//std::cout << "uMax, uMin: " << uMax << "   " << uMin << std::endl;

		std::cout << i<< "   " << uMin / uMax << std::endl;

	}

	


	cv::imshow("SRC", src_8uc1_img); // display image
	cv::imshow("img1", img1); // display image
	cv::imshow("indexation", indexace);

}

int perimeter(cv::Mat indexace, int index)
{
	int per = 0;
	for (int x = 1; x < indexace.rows-1; x++)
	{
		for (int y = 1; y < indexace.cols-1; y++)
		{
			if (indexace.at<uchar>(x, y) == index)
			{
				if (indexace.at<uchar>(x + 1, y) != index) per++;
				if (indexace.at<uchar>(x - 1, y) != index) per++;
				if (indexace.at<uchar>(x, y-1) != index) per++;
				if (indexace.at<uchar>(x, y+1) != index) per++;
			}
		}
	}
	return per;
}
typedef struct obj{
	float F1;
	float F2;
	std::string name;
};

std::string recognizeObject(float F1, float F2, obj* objects)
{
	int index = 0;
	float min = 100;
	for (int i = 0; i < 3; i++)
	{
		if (sqrt(pow(F1 - objects[i].F1, 2) + pow(F2 - objects[i].F2, 2))<min)
		{
			min = sqrt(pow(F1 - objects[i].F1, 2) + pow(F2 - objects[i].F2, 2));
			index = i;
		}
	}

	return objects[index].name;
}

void recognizePicture(cv::Mat indexace, int index, obj * objects, cv::Mat image)
{
	float xt, yt;
	float uMax, uMin;
	for (int i = 1; i < index; i++)
	{
		xt = (float)m(indexace, 1, 0, i) / m(indexace, 0, 0, i);
		yt = (float)m(indexace, 0, 1, i) / m(indexace, 0, 0, i);
		//std::cout << "xt,yt: "<< xt << "   " << yt << std::endl;
		uMax = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) + 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)), 2));
		uMin = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) - 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)), 2));

		float F1 = pow((float)perimeter(indexace, i), 2) / (100 * u(indexace, 0, 0, xt, yt, i));
		float F2 = uMin / uMax;
		std::cout << i << "   " << uMin / uMax << std::endl;
		std::cout << "perimeter: " << perimeter(indexace, i) << "   " << 100 * u(indexace, 0, 0, xt, yt, i) << "   " << F2 << std::endl;


		std::cout << recognizeObject(F1, F2, objects) << std::endl;
		cv::putText(image, recognizeObject(F1, F2, objects), cv::Point(xt, yt), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));

	}
}

void cv6()
{
	cv::Mat src_8uc1_img;
	cv::Mat img1;
	cv::Mat img2;
	cv::Mat indexace;
	cv::Mat indexace2;
	uchar tr = 120;
	obj * objects = new obj[3];
	cv::Mat testImage;

	for (int i = 0; i < 3; i++)
	{
		objects[i].F1 = 0;
		objects[i].F2 = 0;
	}
	objects[0].name = "square";
	objects[1].name = "star";
	objects[2].name = "rectangle";

	src_8uc1_img = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale
	testImage = cv::imread("images/test01.png", CV_LOAD_IMAGE_GRAYSCALE); // load image in grayscale

	img1 = src_8uc1_img.clone();
	img2 = testImage.clone();
	indexace = src_8uc1_img.clone();
	indexace2 = testImage.clone();

	for (int x = 0; x < src_8uc1_img.rows; x++)
	{
		for (int y = 0; y < src_8uc1_img.cols; y++)
		{
			indexace.at<uchar>(x, y) = 0;
			if (src_8uc1_img.at<uchar>(x, y)>tr)
			{
				img1.at<uchar>(x, y) = 255;
			}
			else
			{
				img1.at<uchar>(x, y) = 0;
			}

		}
	}

	for (int x = 0; x < testImage.rows; x++)
	{
		for (int y = 0; y < testImage.cols; y++)
		{
			indexace2.at<uchar>(x, y) = 0;
			if (testImage.at<uchar>(x, y)>tr)
			{
				img2.at<uchar>(x, y) = 255;
			}
			else
			{
				img2.at<uchar>(x, y) = 0;
			}

		}
	}

	int index = 1;
	for (int x = 0; x < img1.rows; x++)
	{
		for (int y = 0; y < img1.cols; y++)
		{
			if (indexation(img1, indexace, x, y, index))
			{
				index++;
			}
		}
	}

	int index2 = 1;
	for (int x = 0; x < img2.rows; x++)
	{
		for (int y = 0; y < img2.cols; y++)
		{
			if (indexation(img2, indexace2, x, y, index2))
			{
				index2++;
			}
		}
	}

	float xt = 0, yt = 0;
	float test;
	float uMin = 0, uMax = 0;

	float dF1 = 0;
	float dF2 = 0;

	for (int i = 1; i < index; i++)
	{
		xt = (float)m(indexace, 1, 0, i) / m(indexace, 0, 0, i);
		yt = (float)m(indexace, 0, 1, i) / m(indexace, 0, 0, i);
		//std::cout << "xt,yt: "<< xt << "   " << yt << std::endl;


		uMax = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) + 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)), 2));
		uMin = 0.5 * (u(indexace, 2, 0, xt, yt, i) + u(indexace, 0, 2, xt, yt, i)) - 0.5*sqrt(4 * u(indexace, 1, 1, xt, yt, i)*u(indexace, 1, 1, xt, yt, i) + pow((u(indexace, 2, 0, xt, yt, i) - u(indexace, 0, 2, xt, yt, i)), 2));

		if (i<5)
		{
			objects[0].F1 += pow((float)perimeter(indexace, i), 2) / (100 * u(indexace, 0, 0, xt, yt, i));
			objects[0].F2 += uMin / uMax;
		} else if (i<9)
		{
			objects[1].F1 += pow((float)perimeter(indexace, i), 2) / (100 * u(indexace, 0, 0, xt, yt, i));
			objects[1].F2 += uMin / uMax;
		} else
		{
			objects[2].F1 += pow((float)perimeter(indexace, i), 2) / (100 * u(indexace, 0, 0, xt, yt, i));
			objects[2].F2 += uMin / uMax;
		}
		//std::cout << "uMax, uMin: " << uMax << "   " << uMin << std::endl;

	}

	for (int i = 0; i < 3; i++)
	{
		objects[i].F1 /= 4;
		objects[i].F2 /=4;
		std::cout << objects[i].F1 << "  " << objects[i].F2 << "   " << objects[i].name << std::endl;
	}

	recognizePicture(indexace,index,objects,src_8uc1_img);
	recognizePicture(indexace,index,objects,testImage);




	cv::imshow("SRC", src_8uc1_img); // display image
	cv::imshow("img1", img1); // display image
	cv::imshow("indexation", indexace);
	cv::imshow("testImage", testImage);

}


int main(int argc, char* argv[])
{
	//testovaci comment
	//cv1();	// detekce hrany
	//cv2();	// lambert + gaus
	//cv3();	// double thresholding
	//cv4();	// indexing
	//cv5();		// mediator
	cv6();		// rozpoznani

	cv::waitKey(0); // press any key to exit
	return 0;
}
