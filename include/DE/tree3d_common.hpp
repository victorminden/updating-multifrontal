/*
   Name:   tree3d_common.hpp
   Author: Victor Minden
   Purpose: The common includes needed for multifrontal and HIF
   Comments: (none)
 */
#pragma once
#ifndef TREEFACTOR3D_COMMON_HPP
#define TREEFACTOR3D_COMMON_HPP 1

#include "linalg.hpp"
#include <set>
#define LINE "--------------------------------------------------------------------------------"

namespace TreeFactor3D {
	typedef enum FaceType {XNORMAL, YNORMAL, ZNORMAL} FaceType;


	class TensorIdx3D {
		public:
			TensorIdx3D() {}
			TensorIdx3D(int intCoord, int width, int height, int depth);
			TensorIdx3D(int x, int y, int z, int width, int height, int depth) : x_(x), y_(y), z_(z), Nx_(width), Ny_(height), Nz_(depth) {}
			int LinearIndex(int width=0, int height=0, int depth=0);

			void print() {
				std::cout << '(' << x_ << ','<<y_ << ',' << z_ << ')' << std::endl;
			}

			inline int x() {return x_;}
			inline int y() {return y_;}
			inline int z() {return z_;}
		private:
			int x_, y_, z_, Nx_, Ny_, Nz_;
	}; // class TensorIdx3D

	TensorIdx3D::TensorIdx3D(int intCoord, int width, int height, int depth) : Nx_(width), Ny_(height), Nz_(depth) {
		x_ = intCoord % width;
		intCoord /= width;
	    y_ = intCoord % height;
	    intCoord /= height;
	    z_ = intCoord;
	    #ifdef DEBUG
	    assert(x_ >= 0 && x_ < width && y_ >=0 && y_ < height && z_ < depth);
	    #endif
	}

	int TensorIdx3D::LinearIndex(int width, int height, int depth) {
		if (width == 0) {
			width  = Nx_;
			height = Ny_;
			depth  = Nz_;
		}
		int intCoord = x_ + y_ * width + z_ * width * height;
		#ifdef DEBUG
		assert(intCoord >=0 && intCoord <= width * height * depth);
		#endif
		return intCoord;
	}


	/* Get the "face neighbors" of a box on a given level */
	IdxVec getNborFacesOfBox_(int boxIdx, int ell);
	/* Get the "box neighbors" of a face on a given level */
	IdxVec getNborBoxesOfFace_(int faceIdx, int ell);
	/* Get the "face neighbors" of a face on a given level */
	IdxVec getNborFacesOfFace_(int faceIdx, int ell);




	/* Find the child faces of a face */
	IdxVec getChildIdxFromFaceIdx_(int idx, int ell);
	/* Find the child boxes of a box */
	IdxVec getChildIdxFromBoxIdx_(int idx, int ell);
	/* Find the parent of a box */
	int getParentIdxFromBoxIdx_(int idx, int ell);

	/* Determine whether a face is x-normal, y-normal, or z-normal */
	inline FaceType getFaceType_(int FaceIdx, int ell);

	/* Map faces from their three dimensional coordinates to their linear index */
	inline int getFaceIdxFromCoords_(int row, int col, int sli, int ell, FaceType type);

	/* Map faces from their linear index to their three dimensional coordinates */
	inline TensorIdx3D getFaceCoordsFromIdx_(int idx, int ell);

	/* Get the "face children" of a box that contain DOFs interior to the box */
	IdxVec getSelfFaceChildrenOfBox_(int boxIdx, int ell);

	/* Get the "face children" of a box that contain DOFs that belong to parent faces */
	IdxVec getSharedFaceChildrenOfBox_(int boxIdx, int ell);



}; // namespace TreeFactor3D

#include "../../src/DE/tree3d_common.cpp"
#endif //ifndef TREEFACTOR3D_COMMON_HPP