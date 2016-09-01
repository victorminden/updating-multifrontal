/*
   Name:   tree2d_common.hpp
   Author: Victor Minden
   Purpose: The common includes needed for multifrontal and HIF
   Comments: (none)
 */
#pragma once
#ifndef TREEFACTOR2D_COMMON_HPP
#define TREEFACTOR2D_COMMON_HPP 1

#include "linalg.hpp"
#include <set>
#define LINE "--------------------------------------------------------------------------------"

namespace TreeFactor2D {
	typedef enum EdgeType {L2R, T2B} EdgeType;


	class TensorIdx2D {
		public:
			TensorIdx2D() {}
			TensorIdx2D(int intCoord, int width, int height);
			TensorIdx2D(int x, int y, int width, int height) : x_(x), y_(y), Nx_(width), Ny_(height) {}
			int LinearIndex(int width=0, int height=0);

			inline int x() {return x_;}
			inline int y() {return y_;}
		private:
			int x_, y_, Nx_, Ny_;
	}; // class TensorIdx2D

	TensorIdx2D::TensorIdx2D(int intCoord, int width, int height) : Nx_(width), Ny_(height) {
		x_ = intCoord % width;
	    y_ = (intCoord - x_) / width;
	    #ifdef DEBUG
	    assert(x_ >= 0 && x_ < width && y_ >=0 && y_ < height);
	    #endif
	}

	int TensorIdx2D::LinearIndex(int width, int height) {
		if (width == 0) {
			width = Nx_;
			height = Ny_;
		}
		int intCoord = x_ + y_ * width;
		#ifdef DEBUG
		assert(intCoord >=0 && intCoord <= width * height);
		#endif
		return intCoord;
	}


	/* Get the "edge neighbors" of a box on a given level */
	IdxVec getNborEdgesOfBox_(int boxIdx, int ell);
	/* Get the "box neighbors" of an edge on a given level */
	IdxVec getNborBoxesOfEdge_(int edgeIdx, int ell);
	/* Get the "edge neighbors" of an edge on a given level */
	IdxVec getNborEdgesOfEdge_(int edgeIdx, int ell);



	/* Find the child edges of an edge */
	IdxVec getChildIdxFromEdgeIdx_(int idx, int ell);
	/* Find the child boxes of an box */
	IdxVec getChildIdxFromBoxIdx_(int idx, int ell);
	/* Find the parent of an box */
	int getParentIdxFromBoxIdx_(int idx, int ell);

	/* Determine whether an edge is left-to-right or top-to-bottom */
	inline EdgeType getEdgeType_(int edgeIdx, int ell);

	/* Map edges from their two dimensional coordinates to their linear index */
	inline int getEdgeIdxFromCoords_(int row, int col, int ell, EdgeType type);

	/* Map edges from their linear index to their two dimensional coordinates */
	inline TensorIdx2D getEdgeCoordsFromIdx_(int idx, int ell);

	/* Get the "edge children" of a box that contain DOFs interior to the box */
	IdxVec getSelfEdgeChildrenOfBox_(int boxIdx, int ell);

	/* Get the "edge children" of a box that contain DOFs that belong to parent edges */
	IdxVec getSharedEdgeChildrenOfBox_(int boxIdx, int ell);



}; // namespace TreeFactor2D

#include "../../src/DE/tree2d_common.cpp"
#endif //ifndef TREEFACTOR2D_COMMON_HPP