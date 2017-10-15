# /****************************************************************************
#
# - Codename:  Image Completion with Intrinsic Reflectance Guidance (BMVC 2017)
#
# - Writers:   Soomin Kim (soo.kim813@gmail.com)
#
# - Institute: KAIST SGVR Lab
#
# - Bibtex:
#
# @InProceedings{Inpainting:BMVC:2017,
#  author  = {Soomin Kim and Taeyoung Kim and Min H. Kim and Sung-Eui Yoon},
#  title   = {Image Completion with Intrinsic Reflectance Guidance},
#  booktitle = {Proc. British Machine Vision Conference (BMVC 2017)},
#  address = {London, England},
#  year = {2017},
#  pages = {},
#  volume  = {},
#  }
#
#
# - License:  GNU General Public License Usage
#   Alternatively, this file may be used under the terms of the GNU General
#   Public License version 3.0 as published by the Free Software Foundation
#   and appearing in the file LICENSE.GPL included in the packaging of this
#   file. Please review the following information to ensure the GNU General
#   Public License version 3.0 requirements will be met:
#   http://www.gnu.org/copyleft/gpl.html.
#
#
# *****************************************************************************/
#cmake entry point
cmake_minimum_required(VERSION 2.8)
project(ReflectanceInpainting)
#==================================================
# src
add_executable(ReflectanceInpainting
	RefInpainting.cpp
	RefInpainting.h
	RefrgbInpainting.cpp 
)
#==================================================
#find packages
find_package(OpenCV)
if (OpenCV_FOUND)
  include_directories( ${OpenCV_INCLUDE_DIRS} )
  target_link_libraries (ReflectanceInpainting ${OpenCV_LIBS})
endif (OpenCV_FOUND)
#==================================================
add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNING
)
#==================================================

