/****************************************************************************

- Codename:  Image Completion with Intrinsic Reflectance Guidance (BMVC 2017)

- Writers:   Soomin Kim (soo.kim813@gmail.com)

- Institute: KAIST SGVR Lab

- Bibtex:

@InProceedings{Inpainting:BMVC:2017,
  author  = {Soomin Kim and Taeyoung Kim and Min H. Kim and Sung-Eui Yoon},
  title   = {Image Completion with Intrinsic Reflectance Guidance},
  booktitle = {Proc. British Machine Vision Conference (BMVC 2017)},
  address = {London, England},
  year = {2017},
  pages = {},
  volume  = {},
}

- License:  GNU General Public License Usage
  Alternatively, this file may be used under the terms of the GNU General
  Public License version 3.0 as published by the Free Software Foundation
  and appearing in the file LICENSE.GPL included in the packaging of this
  file. Please review the following information to ensure the GNU General
  Public License version 3.0 requirements will be met:
  http://www.gnu.org/copyleft/gpl.html.


*****************************************************************************/

#pragma once


#include <iostream>
#include <math.h>
/*OPENCV library*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videostab/inpainting.hpp>
#include <time.h>


#define MASK_THD 0.1
//#define PRINT_MIDDLERESULTS 1
#define CENTERINMASK 1

void displayLABMat(cv::Mat a, char *title, cv::Rect ROI);

template <typename T>
void displayMat(cv::Mat a, char *title, cv::Rect ROI){
	T amin, amax;
	cv::minMaxLoc(a(ROI), &amin, &amax);
	cv::imshow(title, (a(ROI)-amin)/(amax-amin));
	//cv::waitKey();
}

template <typename T>
void displayMatres(cv::Mat a, char *title, cv::Rect ROI,int width, int height){
	T amin, amax;
	cv::Mat tmp;
	cv::minMaxLoc(a(ROI), &amin, &amax);
	cv::resize((a(ROI)-amin)/(amax-amin), tmp,cv::Size(width, height));
	cv::imshow(title, tmp);
	//cv::waitKey();
}

void fixDownsampledMaskMat(cv::Mat mask);
void fixDownsampledMaskMatColorMat(cv::Mat mask,cv::Mat color);
void fixDownsampledMaskMatColorMatRef_ColorMat(cv::Mat mask, cv::Mat color, cv::Mat ref_color);

class ReflectanceInpainting{
public:
	void findNearestNeighbor_withRef(cv::Mat nnf, cv::Mat nnferr, cv::Mat ref_nnferr, bool *patch_type, cv::Mat colormat, cv::Mat ref_colormat, cv::Mat colorfmat, cv::Mat ref_colorfmat, cv::Mat maskmat, std::pair<int, int> size, int emiter, int level, int maxlevel);
	void colorVoteLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size);
	void colorVote(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size);
	void doEMIterwithRef(cv::Mat nnf, cv::Mat nnferr, cv::Mat ref_nnferr, bool *patch_type, cv::Mat colormat, cv::Mat ref_colormat, cv::Mat colorfmat, cv::Mat ref_colorfmat, cv::Mat maskmat, std::pair<int, int> size, int num_emiter, cv::Size orig_size, int level, int maxlevel);
	void constructLaplacianPyr(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat &img);
	void constructLaplacianPyrMask(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat mask,cv::Mat &img);
	void constructGaussianPyr(std::vector<cv::Mat> &gpyr, cv::Mat &img);
	void upscaleImages(cv::Mat nnf, cv::Mat nnferr, bool *patch_type,  cv::Mat colorfmat,  cv::Mat dmaskmat,  cv::Mat umaskmat);

public:
	int psz_, minsize_;
	double dwp_, highconfidence_, gamma_, alpha_, beta_;
	double siminterval_; 
	int patchmatch_iter_;
   int rs_iter_;
	int nnfcount_;

	double ratio_; //for variance ratio
};


class budget{
public:
	int locationX;
	int locationY;
	double error;
	//budget() : locationX(0), error(5000){}
	//budget(int locX, double err) :locationX(locX), error(err){}

	budget() : locationX(0), locationY(0), error(5000){}	
	budget(int locY, int locX, double err) :locationY(locY),locationX(locX), error(err){}

};



