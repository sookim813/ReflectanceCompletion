# Image Completion with Intrinsic Reflectance Guidance


Code for 
**Image Completion with Intrinsic Reflectance Guidance (BMVC 2017)**  
*Soomin Kim and Taeyoung Kim and Min H. Kim and Sung-Eui Yoon*

<http://sglab.kaist.ac.kr/Reflectance_ImgCompl/>

Please cite this paper if you use this code in an academic publication.

```
@InProceedings{Inpainting:BMVC:2017,
  author  = {Soomin Kim and Taeyoung Kim and Min H. Kim and Sung-Eui Yoon},
  title   = {Image Completion with Intrinsic Reflectance Guidance},
  booktitle = {Proc. British Machine Vision Conference (BMVC 2017)},
  address = {London, England},
  year = {2017},
  pages = {},
  volume  = {},
}
```

--------------------------------------------------------------------------------------------------


This is the code for *Image Completion with Intrinsic Reflectance Guidance*. Getting a reflectance code is not included. In the paper, we utilize [learn-reflectance code](https://github.com/tinghuiz/learn-reflectance) for getting reflectances.

The image completion code is built upon [Laplacian inpainting (CVPR2016)](https://github.com/kaist-vclab/laplacianinpainting), which is modified to utilize reflectance information in completion.

Dependency
--------------------------------------------------------------------------------------------------

`OpenCV` >= 2.4.10 required.  
`cmake` >= 2.8 required.


How to compile it
--------------------------------------------------------------------------------------------------
```
>> mkdir build
>> cd build
>> ccmake -DCMAKE_BUILD_TYPE=Release ../
>> make
```

How to use it
--------------------------------------------------------------------------------------------------

User need to create an image mask and obtain a reflectance image from conventional intrinsic decomposition(ex. [learn-reflectance code](https://github.com/tinghuiz/learn-reflectance)).
Then, run a program with arguments. There are default options, so you can put four arguments(file names) without specific arguments for demo  
*:a project name, a rgb image, a reflectance image and a mask image name.*

`Syntax example: ReflectanceInpainting.exe man.png man_ref.png man_mask.png`

(Optional) If you want to modify the default options, then you can modify them with following argument. 



`Syntax example: ReflectanceInpainting.exe man.png man_ref.png man_mask.png [opt1] [opt2] [opt3] [opt4] [opt5] [opt6] [opt7] [opt8] [opt9] [opt10] `

```
[opt1] patch size: (default) 7
[opt2] distance weight parameter for recontruction: (default) 1.3
[opt3] minimum size in percentages: (default) 20
[opt4] alpha (initial original color weight): (default) 0.05
[opt5] beta (initial reflectacne color weight): (default) 0.65
[opt6] gamma (original color feature weight): (default) 0.3
[opt7] number of EM: (default) 50\n
[opt8] decrease factor in EM: (default) 10
[opt9] minimum iteration: (default) 10
[opt10] random search iteration: (default) 1
```

For detail of arguments, please see the code and the paper.


