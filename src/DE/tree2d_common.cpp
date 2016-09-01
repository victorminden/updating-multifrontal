#include "tree2d_common.hpp"

namespace TreeFactor2D {
	using namespace LinAlg;
	using namespace std;

	/* Utility function for getting edgeType */
	EdgeType getEdgeType_(int edgeIdx, int ell) {
		int boxesPerSide = 1 << ell;
		return (edgeIdx % (2*boxesPerSide - 1) < boxesPerSide-1)? T2B : L2R;
	}


	/* Utility routines for getting neighbors */
	int getEdgeIdxFromCoords_(int row, int col, int ell, EdgeType type) {
		int boxesPerSide = 1 << ell;
		if (type == L2R) {
			int l2rIdx = row * (2*boxesPerSide - 1) + col + boxesPerSide - 1;
			return l2rIdx;
		} else if (type == T2B) {
			int t2bIdx = row * (2*boxesPerSide - 1) + col;
			return t2bIdx;
		} else {
			throw runtime_error("Unrecognized edge type!");
		}
	}

	TensorIdx2D getEdgeCoordsFromIdx_(int idx, int ell) {
		int boxesPerSide = 1 << ell;
		EdgeType type = getEdgeType_(idx, ell);
		if (type == L2R) {
			idx -= (boxesPerSide - 1);
		}
		return TensorIdx2D(idx,2*boxesPerSide - 1, 2*boxesPerSide - 1);
	}

	IdxVec getSharedEdgeChildrenOfBox_(int boxIdx, int ell) {
		set<int> chldEdges;
		IdxVec chldBoxes = getChildIdxFromBoxIdx_(boxIdx, ell);
		for (vector<int>::iterator it = chldBoxes.begin(); it != chldBoxes.end(); it++) {
			int chldIdx = *it;
			IdxVec edges = getNborEdgesOfBox_(chldIdx, ell+1);
			for (vector<int>::iterator edgeIt = edges.begin(); edgeIt!= edges.end(); edgeIt++) {
				int edgeIdx = *edgeIt;
				if (edgeIdx == -1) {
					continue;
				}
				TensorIdx2D edgeCoords = getEdgeCoordsFromIdx_(edgeIdx, ell+1);
				EdgeType type = getEdgeType_(edgeIdx, ell+1);
				if ((type == T2B && edgeCoords.x() % 2 == 1) || (type == L2R && edgeCoords.y() % 2 == 1)) {
					// exterior edge, include
					chldEdges.insert(edgeIdx);
				}
			}
		}
		return IdxVec(chldEdges.begin(), chldEdges.end());
	}

	IdxVec getSelfEdgeChildrenOfBox_(int boxIdx, int ell) {
		set<int> chldEdges;
		IdxVec chldBoxes = getChildIdxFromBoxIdx_(boxIdx, ell);
		for (vector<int>::iterator it = chldBoxes.begin(); it != chldBoxes.end(); it++) {
			int chldIdx = *it;
			IdxVec edges = getNborEdgesOfBox_(chldIdx, ell+1);
			for (vector<int>::iterator edgeIt = edges.begin(); edgeIt!= edges.end(); edgeIt++) {
				int edgeIdx = *edgeIt;
				if (edgeIdx == -1) {
					continue;
				}
				TensorIdx2D edgeCoords = getEdgeCoordsFromIdx_(edgeIdx, ell+1);
				EdgeType type = getEdgeType_(edgeIdx, ell+1);
				if ((type == T2B && edgeCoords.x() % 2 == 0) || (type == L2R && edgeCoords.y() % 2 == 0)) {
					// interior edge, include
					chldEdges.insert(edgeIdx);
				}
			}
		}
		return IdxVec(chldEdges.begin(), chldEdges.end());
	}



		/* Utility routines for getting parents and children */
	int getParentIdxFromBoxIdx_(int boxIdx, int ell){
		int boxesPerSide = 1 << ell;
		TensorIdx2D boxCoord(boxIdx, boxesPerSide, boxesPerSide);
		TensorIdx2D parentCoord(boxCoord.x() / 2, boxCoord.y() / 2, boxesPerSide/2, boxesPerSide / 2);
		return parentCoord.LinearIndex();
	}

	IdxVec getChildIdxFromEdgeIdx_(int idx, int ell) {

		EdgeType type = getEdgeType_(idx,ell);
		IdxVec childInds;

		TensorIdx2D coord = getEdgeCoordsFromIdx_(idx, ell);

		int row = coord.y();
		int col = coord.x();

		if (type == L2R) {
			// First child has twice my col, same row
			int childIdx1 = getEdgeIdxFromCoords_(2 * row + 1, 2 * col, ell + 1, L2R);
			// Second child has twice my col + 1, same row
			int childIdx2 = getEdgeIdxFromCoords_(2 * row + 1, 2 * col + 1, ell + 1, L2R);
			childInds.push_back(childIdx1);
			childInds.push_back(childIdx2);
			return childInds;
		} else {
			// First child has twice my row, twice my col + 1
			int childIdx1 = getEdgeIdxFromCoords_(2 * row, 2 * col + 1, ell + 1, T2B);
			// Second child has twice my row + 1, same col
			int childIdx2 = getEdgeIdxFromCoords_(2 * row + 1, 2 * col + 1, ell + 1, T2B);
			childInds.push_back(childIdx1);
			childInds.push_back(childIdx2);
			return childInds;
		}

	}

	IdxVec getChildIdxFromBoxIdx_(int idx, int ell) {
		IdxVec childInds;
		int boxesPerSide = 1 << ell;
		TensorIdx2D coord(idx,boxesPerSide, boxesPerSide);

		childInds.push_back(TensorIdx2D(2*coord.x(), 2*coord.y(),2*boxesPerSide,2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx2D(2*coord.x(), 2*coord.y()+1,2*boxesPerSide,2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx2D(2*coord.x()+1, 2*coord.y(),2*boxesPerSide,2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx2D(2*coord.x()+1, 2*coord.y()+1,2*boxesPerSide,2*boxesPerSide).LinearIndex());
		return childInds;
	}


	/* Given the index of a box, find the indices of the edges bordering it  */
	IdxVec getNborEdgesOfBox_(int boxIdx, int ell) {
		int boxesPerSide = 1 << ell;
		TensorIdx2D boxCoord(boxIdx, boxesPerSide, boxesPerSide);

		IdxVec edgeIdxs;
		// Now the T2B (top-to-bottom) edges

		int t2bRow = boxCoord.y();
		if (boxCoord.x() > 0) {
			// This box has a left edge
			int t2bColumn = boxCoord.x() - 1;
			int t2bIdx = getEdgeIdxFromCoords_(t2bRow, t2bColumn, ell, T2B);
			edgeIdxs.push_back(t2bIdx);
		}
		else {
			edgeIdxs.push_back(-1);
		}
		if (boxCoord.x() < boxesPerSide - 1) {
			// This box has a right edge
			int t2bColumn = boxCoord.x();
			int t2bIdx = getEdgeIdxFromCoords_(t2bRow, t2bColumn, ell, T2B);
			edgeIdxs.push_back(t2bIdx);
		}
		else {
			edgeIdxs.push_back(-1);
		}

		// Now the L2R (left-to-right) edges
		int l2rColumn = boxCoord.x(); // the columns for l2r edges line up with boxes
		if (boxCoord.y() > 0) {
			// This box has a bottom edge
			int l2rRow = boxCoord.y() - 1;
			int l2rIdx = getEdgeIdxFromCoords_(l2rRow, l2rColumn, ell, L2R);
			edgeIdxs.push_back(l2rIdx);
		}
		else {
			edgeIdxs.push_back(-1);
		}
		if (boxCoord.y() < boxesPerSide - 1) {
			// This box has a top edge
			int l2rRow = boxCoord.y();
			int l2rIdx = getEdgeIdxFromCoords_(l2rRow, l2rColumn, ell, L2R);
			edgeIdxs.push_back(l2rIdx);
		}
		else {
			edgeIdxs.push_back(-1);
		}

		return edgeIdxs;
	}

	IdxVec getNborBoxesOfEdge_(int idx, int ell) {
		IdxVec nbors;
		int boxesPerSide = 1 << ell;

		TensorIdx2D coord = getEdgeCoordsFromIdx_(idx,ell);
		int col = coord.x();
		int row = coord.y();
		EdgeType type = getEdgeType_(idx,ell);

		if (type == T2B) {
			// Edge only has left and right neighbors

			TensorIdx2D boxCoordL(col, row, boxesPerSide, boxesPerSide);
			TensorIdx2D boxCoordR(col+1, row, boxesPerSide, boxesPerSide);
			nbors.push_back(boxCoordL.LinearIndex());
			nbors.push_back(boxCoordR.LinearIndex());
			return nbors;
		} else {
			// Edge only has up and down neighbors
			TensorIdx2D boxCoordB(col, row, boxesPerSide, boxesPerSide);
			TensorIdx2D boxCoordT(col, row+1, boxesPerSide, boxesPerSide);
			nbors.push_back(boxCoordB.LinearIndex());
			nbors.push_back(boxCoordT.LinearIndex());
			return nbors;
		}
	}

	IdxVec getNborEdgesOfEdge_(int idx, int ell) {
		IdxVec nborBoxes = getNborBoxesOfEdge_(idx, ell);
		IdxVec nbors;
		nbors.reserve(6);
		for (int i = 0; i < nborBoxes.size(); i++) {
			IdxVec nborEdges = getNborEdgesOfBox_(nborBoxes[i], ell);
			for (int j = 0; j < nborEdges.size(); j++) {
				if (nborEdges[j] != idx) {
					nbors.push_back(nborEdges[j]);
				}
			}
		}
		return nbors;
	}



}; // namespace TreeFactor2D