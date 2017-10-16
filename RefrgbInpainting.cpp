/****************************************************************************

- Codename:  Image Completion with Intrinsic Reflectance Guidance (BMVC 2017)

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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <memory>
#include <limits.h>
#include <algorithm>
#include <time.h>

#include "RefInpainting.h"


FILE *logout;
int main(int argc, char **argv){
   
   //Argument description
   //argv[1]   project name
   //argv[2]   rgb image name
   //argv[3]   reflectance image
   //argv[4]   mask image name
   //argv[5]   patch size
   //argv[6]   distance weight parameter. [Wexler et al. 2007]
   //argv[7]   minimum size for resizing
   //argv[8]   initial original color weight
   //argv[9]   initial reflectacne color weight
   //argv[10]  original color feature weight
   //argv[11]   the number of EM iteration
   //argv[12]   decrease factor of EM iteration
   //argv[13]  minimum EM iteration
   //argv[14]  random search iteration

   // You have to give at least four arguments: a project name, a rgb image, reflectance image and an mask image name.
   // Then, you can run our algorithm with default setting.
   // If you want to change parameters, just give -1 for unchanged varaibles and give a number which you want.
   // EX) ReflectanceInpainting.exe man.png man_ref.png man_mask.png  

	cv::Mat maskmat, colormat, ref_colormat, origcolormat, rgbmat, ref_rgbmat; 
	double *colorptr, *ref_colorptr, *maskptr;
	int dheight, dwidth;
	int height, width;
	unsigned char *mask;


	int decrease_factor;
	int min_iter;
	double *scales;
	char *outputfilename, *outputfilename2, *fname, *processfilename, *dirname;
	time_t timer;

	//inpainting parameter
  	int num_scale;
	int num_em;
	int psz;
	int min_size;
   int rs_iter;
	double dwp;
	double gamma;
	double alpha;
	double beta;

   //pyramid
   //gpyr - Gaussian pyramid
   //upyr - upsampled Gaussian pyramid
   //fpyr - Laplacian pyramid
	std::vector<std::pair<int,int> > pyr_size;
	std::vector<cv::Mat> mask_gpyr, color_gpyr, ref_color_gpyr; 
	std::vector<cv::Mat> mask_upyr, color_upyr, ref_color_upyr;
	std::vector<cv::Mat> mask_fpyr, color_fpyr, ref_color_fpyr;
	std::vector<cv::Mat> rgb_gpyr,rgb_fpyr,rgb_upyr;

   //Laplacian inpainting object
	ReflectanceInpainting inpainting;

	//logout = fopen("output.txt","w");//for debug

	processfilename = (char*)malloc(sizeof(char) * 200);
	dirname = (char*)malloc(sizeof(char) *200);
	fname = (char*)malloc(sizeof(char) *200);
	outputfilename = (char*)malloc(sizeof(char) *200);
	outputfilename2 = (char*)malloc(sizeof(char)* 200);

 
   ////////////////////////
	//*Step 1: read input*//
	////////////////////////
   
   if(argc > 4) {
      colormat = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);     //read a rgb image
	  ref_colormat = cv::imread(argv[3], CV_LOAD_IMAGE_COLOR); //read a reflectance image
	   maskmat = cv::imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE);  //read a mask image
   }
   else{
//      printf("No image and no mask.\n"); 
	   printf("Image Completion with Intrinsic Reflectance Guidance (BMVC 2017) ver. 1.0\n");
	   printf("Copyright (c) 2016 Joo Ho Lee, Min H. Kim\n"); 
	   printf("\n"); 
	   printf("Syntax example: ReflectanceInpainting.exe man.png man_ref.png man_mask.png [opt1] [opt2] [opt3] [opt4] [opt5] [opt6] [opt7]\n"); 
	   printf("\n"); 
	   printf("[opt1] patch size: (default) 7\n");
	   printf("[opt2] distance weight parameter for recontruction: (default) 1.3\n");
	   printf("[opt3] minimum size in percentages: (default) 20\n");
	   printf("[opt4] alpha (initial original color weight): (default) 0.05 \n");
	   printf("[opt5] beta (initial reflectacne color weight): (default) 0.65\n");
	   printf("[opt6] gamma (original color feature weight): (default) 0.3\n");
	   printf("[opt7] number of EM: (default) 50\n");
	   printf("[opt8] decrease factor in EM: (default) 10\n");
	   printf("[opt9] minimum iteration: (default) 10\n");
	   printf("[opt10] random search iteration: (default) 1\n"); 
	   printf("\n"); 
       return 1;
   }

   psz            = (argc<6 || atoi(argv[5])  == -1) ? 7  : atoi(argv[5]);  //patch size
	dwp          = (argc<7 || atof(argv[6])  == -1) ? 1.3: atof(argv[6]);//distance weight parameter
	min_size       = (argc<8 || atoi(argv[7])  == -1) ? 20 : atoi(argv[7]);  //minimum size
	alpha			= (argc<9 || atoi(argv[8]) == -1) ? 0.05 : atof(argv[8]);//alpha - initial original color weight         
	beta			= (argc<10 || atoi(argv[9]) == -1) ? 0.65 : atof(argv[9]);//beta - initial reflectance color weight
	gamma			= (argc<11 || atoi(argv[10]) == -1) ? 0.3 : atof(argv[10]);//gamma - weight of image color feature
	num_em         = (argc<12 || atoi(argv[11])  == -1) ? 50 : atoi(argv[11]); //the number of EM iteration
	decrease_factor= (argc<13|| atoi(argv[12])  == -1) ? 10 : atoi(argv[12]); //decrease_factor
	min_iter       = (argc<14|| atoi(argv[13]) == -1) ? 10 : atoi(argv[13]);//minimum iteration
   rs_iter        = (argc<15|| atoi(argv[14]) == -1) ? 1  : atoi(argv[14]); //random search iteration //
     
	width = colormat.cols;  //image width
	height = colormat.rows; //image height

   int tmp_width = width,tmp_height = height;
   int tmp = 1;
   for(int i=0;;i++){
      tmp_width  >>= 1;
      tmp_height >>= 1;
      if(min_size > tmp_width || min_size > tmp_height)
         break;
      tmp <<= 1;
   }

	if(width%tmp) width=width-(width%tmp);
	if(height%tmp) height=height-(height%tmp);
   
	//before crop the img, calculate std of the img and ref.
	cv::Scalar meanImg, devImg, meanRef, devRef;
	cv::meanStdDev(colormat, meanImg, devImg); //if we input last argument as mask, then it will calculate elsewhere
	cv::meanStdDev(ref_colormat, meanRef, devRef);

	double M_img = meanImg.val[0];
	double D_img = devImg.val[0];
	double M_ref = meanRef.val[0];
	double D_ref = devRef.val[0];
	inpainting.ratio_ = D_ref / D_img;

	printf("Img mean: %f, Img dev:%f and Ref mean: %f, Ref dev: %f \n", M_img, D_img, M_ref, D_ref);
	printf("Ref_dev/Img_dev %f \n", inpainting.ratio_);

   //origcolormat = colormat.clone();  //unecessary?

	colormat = colormat(cv::Rect(0,0,width, height));  //crop the image
	ref_colormat = ref_colormat(cv::Rect(0, 0, width, height));
	maskmat = maskmat(cv::Rect(0,0,width, height));

	colormat.convertTo(colormat, CV_32FC3);   //convert an uchar image to a float image (Input of cvtColor function should be a single precision )
	ref_colormat.convertTo(ref_colormat, CV_32FC3); // convert reflectance layer as well
	maskmat.convertTo(maskmat,CV_64FC1);      //double mask

	colormat/=255.0;	//255 -> 1.0
	ref_colormat /= 255.0;
	maskmat/=255.0;

	colormat.convertTo(rgbmat, CV_64FC3);
	ref_colormat.convertTo(ref_rgbmat, CV_64FC3);

	//convert rgb to CIEL*a*b*
	cvtColor(colormat, colormat, CV_RGB2Lab); //RGB to Lab
	colormat.convertTo(colormat, CV_64FC3);   //single -> double

	cvtColor(ref_colormat, ref_colormat, CV_RGB2Lab); //REF  :   RGB to Lab
	ref_colormat.convertTo(ref_colormat, CV_64FC3);   //REF  :   single -> double

	//values in mask region should be zero.
	colorptr = (double*) colormat.data;
	ref_colorptr = (double*)ref_colormat.data;
	maskptr = (double*) maskmat.data;
	
   //refine mask and color image
   for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			int ndx = i*width + j;
			if(maskptr[ndx]>0){
				colorptr[3*ndx] = 0;
				colorptr[3*ndx+1] = 0;
				colorptr[3*ndx+2] = 0;

				ref_colorptr[3 * ndx] = 0;    // make reflectance layer's hole is 0 as well
				ref_colorptr[3 * ndx + 1] = 0;
				ref_colorptr[3 * ndx + 2] = 0;


				maskptr[ndx]=1;
			}
			else maskptr[ndx]=0;
		}
	}
   
   ///////////////////////////////////
	//*step 2: set parameters       *//
   ///////////////////////////////////

	inpainting.dwp_=dwp;                //parameter for voting
	inpainting.alpha_ = alpha;		    // original color weight
	inpainting.beta_ = beta;		    // reflectance color weight
	inpainting.gamma_ = gamma;          //weight of image color feature. 
	inpainting.minsize_=min_size;       //minimum scale
	inpainting.psz_=psz;                //patch size
	inpainting.highconfidence_= 1.0f;   //confidence for non-mask region
	inpainting.patchmatch_iter_ = 12;   //EM iteration
	inpainting.siminterval_ = 3.0f;     //parameter for voting
   inpainting.rs_iter_ = rs_iter;      //random search itertation
  

   sprintf(fname, "RefInpainting_%s_psz%02d_alpha%.2f_beta%.2f_gamma%.2f_minsize%02d_simint_%.1f", argv[1], inpainting.psz_, inpainting.alpha_, inpainting.beta_, inpainting.gamma_, inpainting.minsize_, inpainting.siminterval_);

   ///////////////////////////////////
   //*step 3: generate pyramid     *//
   ///////////////////////////////////

   //inpainting.constructLaplacianPyr(rgb_gpyr, rgb_upyr, rgb_fpyr, rgbmat); // for test? will not be used

	//construct Laplacian pyramid
	inpainting.constructLaplacianPyr(color_gpyr, color_upyr, color_fpyr, colormat);
	inpainting.constructLaplacianPyr(ref_color_gpyr, ref_color_upyr, ref_color_fpyr, ref_colormat);
	inpainting.constructLaplacianPyr(mask_gpyr, mask_upyr, mask_fpyr, maskmat);

	//reverse order (from low-res to high-res)
	std::reverse(color_gpyr.begin(), color_gpyr.end());
	std::reverse(color_upyr.begin(), color_upyr.end());
	std::reverse(color_fpyr.begin(), color_fpyr.end());

	std::reverse(ref_color_gpyr.begin(), ref_color_gpyr.end());
	std::reverse(ref_color_upyr.begin(), ref_color_upyr.end());
	std::reverse(ref_color_fpyr.begin(), ref_color_fpyr.end());

	std::reverse(mask_gpyr.begin(), mask_gpyr.end());
	std::reverse(mask_upyr.begin(), mask_upyr.end());
	std::reverse(mask_fpyr.begin(), mask_fpyr.end());


	//compute pyr_size
   pyr_size.clear();
	
   //set size
   for(int i=0;i<color_gpyr.size();i++){
		pyr_size.push_back(std::pair<int,int>(color_gpyr[i].rows, color_gpyr[i].cols));
		printf("%dth image size: %d %d\n", i,color_gpyr[i].rows,color_gpyr[i].cols);
	}

   //print reflectance here for figure
  // displayMat<double>(ref_color_fpyr[3], "ref_feature", cv::Rect(0, 0, ref_color_fpyr[3].cols, ref_color_fpyr[3].rows));
  // displayLABMat(ref_trg_color, "ref_color", cv::Rect(0, 0, ref_trg_color.cols, ref_trg_color.rows));
 // displayMat<double>(color_fpyr[3], "feature", cv::Rect(0, 0, color_fpyr[3].cols, color_fpyr[3].rows));
   //displayLABMat(trg_color, "color", cv::Rect(0, 0, trg_color.cols, trg_color.rows));
   //cv::waitKey();


   //refine mask
   fixDownsampledMaskMatColorMatRef_ColorMat(mask_gpyr[0], color_gpyr[0], ref_color_gpyr[0]);

   for (int i = 0; i<mask_upyr.size(); i++){
	   fixDownsampledMaskMatColorMatRef_ColorMat(mask_upyr[i], color_upyr[i], ref_color_upyr[i]);
	   fixDownsampledMaskMatColorMatRef_ColorMat(mask_gpyr[i + 1], color_gpyr[i + 1], ref_color_gpyr[i+1]);
	   color_fpyr[i] = color_gpyr[i + 1] - color_upyr[i];
	   ref_color_fpyr[i] = ref_color_gpyr[i + 1] - ref_color_upyr[i];


	   mask_upyr[i] = mask_gpyr[i + 1] + mask_upyr[i];
	   fixDownsampledMaskMat(mask_upyr[i]);
	   fixDownsampledMaskMatColorMatRef_ColorMat(mask_upyr[i], color_upyr[i], ref_color_upyr[i]);
	   fixDownsampledMaskMatColorMatRef_ColorMat(mask_upyr[i], color_gpyr[i + 1], ref_color_gpyr[i+1]);

   }
	//dilate mask

   /////////////////////////////////////////////
	//*step 4: initialize the zero level image*//
   /////////////////////////////////////////////

	cv::Mat color8u, mask8u, feature8u, ref_color8u, ref_feature8u;
	//cv::Mat repmask;
	cv::Mat trg_color, ref_trg_color;
	cv::Mat trg_feature, ref_trg_feature;

	double featuremin, featuremax;
	cv::minMaxLoc(color_fpyr[0], &featuremin, &featuremax);

	//if you don't use reflectance's feature, below is not necessary
	double ref_featuremin, ref_featuremax;
	cv::minMaxLoc(ref_color_fpyr[0], &ref_featuremin, &ref_featuremax);

	color_upyr[0].convertTo(color8u,CV_32FC3);
	ref_color_upyr[0].convertTo(ref_color8u, CV_32FC3);

	cvtColor(color8u, color8u, CV_Lab2RGB); // Lab to RGB
	cvtColor(ref_color8u, ref_color8u, CV_Lab2RGB); // same to Reflectance layer

	color8u = color8u*255.;
	ref_color8u = ref_color8u*255.;

	mask8u = mask_upyr[0]*255.;
	
	//for display
	//color8u.convertTo(color8u, CV_8UC3);
	//ref_color8u.convertTo(ref_color8u, CV_8UC3);
	//cv::imshow("img", color8u);
	//cv::imshow("reflectance", ref_color8u);
	//cv::imshow("asdf",mask8u);
		//cv::waitKey();

		

	feature8u = (color_fpyr[0]-featuremin)/(featuremax-featuremin) * 255.;
	ref_feature8u = (ref_color_fpyr[0] - ref_featuremin) / (ref_featuremax - ref_featuremin)*255.;

	color8u.convertTo(color8u, CV_8U);
	ref_color8u.convertTo(ref_color8u, CV_8U);
	mask8u.convertTo(mask8u, CV_8U);
	feature8u.convertTo(feature8u, CV_8U);
	ref_feature8u.convertTo(ref_feature8u, CV_8U);

	
   //initialization
   //We use a Navier-Stokes based method [Navier et al. 01] only for initialization.
   //http://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf
	cv::inpaint(color8u, mask8u, color8u, 10, 0);
	cv::inpaint(feature8u, mask8u, feature8u, 10, 0);

	cv::inpaint(ref_color8u, mask8u, ref_color8u, 10, 0); //contain inpaint radius 
	cv::inpaint(ref_feature8u, mask8u, ref_feature8u, 10, 0); //

	color8u.convertTo(color8u,CV_32FC3);
	color8u=color8u/255.f;
	cvtColor(color8u, color8u, CV_RGB2Lab);
	color8u.convertTo(color_upyr[0],CV_64FC3);
	feature8u.convertTo(color_fpyr[0],CV_64FC3);
	color_fpyr[0] = color_fpyr[0]/255.0 * (featuremax-featuremin) + featuremin;
	

	ref_color8u.convertTo(ref_color8u, CV_32FC3);
	ref_color8u = ref_color8u / 255.f;
	cvtColor(ref_color8u, ref_color8u, CV_RGB2Lab);
	ref_color8u.convertTo(ref_color_upyr[0], CV_64FC3);
	ref_feature8u.convertTo(ref_color_fpyr[0], CV_64FC3);
	ref_color_fpyr[0] = ref_color_fpyr[0] / 255.0 * (ref_featuremax - ref_featuremin) + ref_featuremin;


	trg_color = color_upyr[0].clone();
	trg_feature = color_fpyr[0].clone();

	// for reflectance layer
	ref_trg_color = ref_color_upyr[0].clone();
	ref_trg_feature = ref_color_fpyr[0].clone();

	
	//displayMat<double>(ref_trg_feature, "ref_feature", cv::Rect(0, 0, ref_trg_feature.cols, ref_trg_feature.rows));
	//displayLABMat(ref_trg_color,"ref_color",cv::Rect(0,0,ref_trg_color.cols, ref_trg_color.rows));
	//displayMat<double>(trg_feature,"feature",cv::Rect(0,0,trg_feature.cols, trg_feature.rows));
	//displayLABMat(trg_color,"color",cv::Rect(0,0,trg_color.cols, trg_color.rows));

	//cv::waitKey();
	
	

	int cur_iter = num_em;

   /////////////////////////////////
	//*Step 5: Do image completion*//
   /////////////////////////////////


	cv::Mat nnf, nnff;
	cv::Mat nnferr, ref_nnferr;
	cv::Mat nxt_color, ref_nxt_color;
	bool *patch_type = NULL;

	nnf = cv::Mat::zeros(pyr_size[1].first, pyr_size[1].second, CV_32SC2); // H x W x 2 int

	clock_t t;
	clock_t recont,accumt;
	int f;
	accumt=0;
	t=clock();


	for(int ilevel = 0; ilevel < color_upyr.size(); ilevel++){
		printf("Processing %dth scale image\n", ilevel);

		if(ilevel){ // ilevel is over 0 (from 1~pyrd level)
			printf("in the ilevle\n");
			//resize trg_color, trg_depth, trg_feature
			recont = clock();
			nxt_color = trg_color + trg_feature; //Gaussian = upsampled Gaussian + Laplacian
			ref_nxt_color = ref_trg_color + ref_trg_feature; // reflectance as well

			recont = clock()-recont;
			accumt+=recont;

			cv::pyrUp(nxt_color, trg_color, cv::Size(trg_color.cols*2, trg_color.rows*2)); // upsample a low-level Gaussian image
			cv::pyrUp(trg_feature, trg_feature, cv::Size(trg_feature.cols*2, trg_feature.rows*2)); //upsample a Laplacian image (we will reset a initial laplacian image later)
			//for reflectance layer
			cv::pyrUp(ref_nxt_color, ref_trg_color, cv::Size(ref_trg_color.cols * 2, ref_trg_color.rows * 2));
			cv::pyrUp(ref_trg_feature, ref_trg_feature, cv::Size(ref_trg_feature.cols * 2, ref_trg_feature.rows * 2));

 
			double *trgcptr = (double*) trg_color.data;
			double *trgfptr = (double*) trg_feature.data;
			double *ref_trgcptr = (double*)ref_trg_color.data;
			double *ref_trgfptr = (double*)ref_trg_feature.data;

			double *maskptr = (double*) mask_upyr[ilevel].data;
			int *nnfptr = (int*) nnf.data;

			//initialize
			for(int i=0;i<pyr_size[ilevel+1].first;i++){
				for(int j=0;j<pyr_size[ilevel+1].second;j++){
					int ndx = i * pyr_size[ilevel+1].second + j;
					if(maskptr[ndx]<0.1){
						trgcptr[3*ndx] = ((double*)(color_upyr[ilevel].data))[3*ndx];
						trgcptr[3*ndx+1] = ((double*)(color_upyr[ilevel].data))[3*ndx+1];
						trgcptr[3*ndx+2] = ((double*)(color_upyr[ilevel].data))[3*ndx+2]; 
						trgfptr[3*ndx] = ((double*)(color_fpyr[ilevel].data))[3*ndx];
						trgfptr[3*ndx+1] = ((double*)(color_fpyr[ilevel].data))[3*ndx+1];
						trgfptr[3*ndx+2] = ((double*)(color_fpyr[ilevel].data))[3*ndx+2]; 

						ref_trgcptr[3 * ndx] = ((double*)(ref_color_upyr[ilevel].data))[3 * ndx];
						ref_trgcptr[3 * ndx + 1] = ((double*)(ref_color_upyr[ilevel].data))[3 * ndx + 1];
						ref_trgcptr[3 * ndx + 2] = ((double*)(ref_color_upyr[ilevel].data))[3 * ndx + 2];
						ref_trgfptr[3 * ndx] = ((double*)(ref_color_fpyr[ilevel].data))[3 * ndx];
						ref_trgfptr[3 * ndx + 1] = ((double*)(ref_color_fpyr[ilevel].data))[3 * ndx + 1];
						ref_trgfptr[3 * ndx + 2] = ((double*)(ref_color_fpyr[ilevel].data))[3 * ndx + 2];

					}
				}
			}

         //NNF propagation
			recont = clock();
         inpainting.upscaleImages(nnf, nnferr, patch_type, trg_feature, mask_upyr[ilevel-1].clone(), mask_upyr[ilevel].clone());
		 inpainting.upscaleImages(nnf, ref_nnferr, patch_type, ref_trg_feature, mask_upyr[ilevel - 1].clone(), mask_upyr[ilevel].clone()); //same to reflection laplacian as well
			recont = clock() - recont;
			accumt += recont;

         //upscale NNF field
			nnf.convertTo(nnff,CV_64FC2);
			cv::resize(nnff, nnff, cv::Size(pyr_size[ilevel+1].second, pyr_size[ilevel+1].first),cv::INTER_LINEAR);
			nnff.convertTo(nnf,CV_32SC2);
			nnff = nnf * 2;
		}

      if(patch_type != NULL)
   		free(patch_type);
		patch_type = (bool*) malloc(sizeof(bool) * pyr_size[ilevel+1].first * pyr_size[ilevel+1].second); 

		nnferr = cv::Mat::zeros(pyr_size[ilevel+1].first, pyr_size[ilevel+1].second, CV_64FC1); // H x W x 1 double
		ref_nnferr = cv::Mat::zeros(pyr_size[ilevel + 1].first, pyr_size[ilevel + 1].second, CV_64FC1); // H x W x 1 double for ref_layer's error too

		//do EM iteration
		sprintf(processfilename, "%s_scale%02d", fname, ilevel); 
		
		inpainting.doEMIterwithRef(nnf, nnferr, ref_nnferr, patch_type, trg_color, ref_trg_color, trg_feature, ref_trg_feature, mask_upyr[ilevel].clone(), pyr_size[ilevel + 1], cur_iter, cv::Size(width, height), ilevel+1, color_upyr.size());
		
		//compute next iteration
		cur_iter -= decrease_factor;
		if(cur_iter<min_iter)
			cur_iter=min_iter;

	}

   //print final result
	cv::Mat tmpimg, ref_tmpimg;	
	sprintf(outputfilename,"%s_final.png", fname); 
	sprintf(outputfilename2, "%s_final_Ref.png", fname);

	tmpimg = trg_color.clone() + trg_feature.clone();
	tmpimg.convertTo(tmpimg, CV_32FC3);
	cvtColor(tmpimg, tmpimg, CV_Lab2RGB);
	tmpimg=255*tmpimg;
	tmpimg.convertTo(tmpimg, CV_8UC3);

	ref_tmpimg = ref_trg_color.clone() + ref_trg_feature.clone();
	ref_tmpimg.convertTo(ref_tmpimg, CV_32FC3);
	cvtColor(ref_tmpimg, ref_tmpimg, CV_Lab2RGB);
	ref_tmpimg = 255 * ref_tmpimg;
	ref_tmpimg.convertTo(ref_tmpimg, CV_8UC3);


	cv::imwrite(outputfilename, tmpimg);
	cv::imwrite(outputfilename2, ref_tmpimg);


	t = clock() - t;
	//printf ("It took %d clicks (%f seconds, (%d,%f) for reconstruction).\n",(int)t,((float)t)/CLOCKS_PER_SEC, (int)accumt, (float)accumt/CLOCKS_PER_SEC);
	float total_secs = ((float)t)/((float)CLOCKS_PER_SEC);
	int mins = (int)floor(total_secs/60.f);
	int secs = (int)total_secs - (mins*60);
	printf ("It took %d:%d (minutes:seconds).\n", mins, secs);

    free(processfilename);
	free(dirname);
	free(fname);
	free(outputfilename);
	free(outputfilename2);


   return 0;
}
