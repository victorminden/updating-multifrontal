#include "tree3d_common.hpp"

namespace TreeFactor3D {
	using namespace LinAlg;
	using namespace std;


	/* Utility function for getting faceType */
	FaceType getFaceType_(int faceIdx, int ell) {
		int boxesPerSide = 1 << ell;
		int facesPerSlice = 2 * (boxesPerSide-1)*boxesPerSide + boxesPerSide * boxesPerSide;
		faceIdx %= facesPerSlice;
		if (faceIdx  >= 2 * (boxesPerSide-1)*boxesPerSide) {
			// Is a znormal face
			return ZNORMAL;
		}
		faceIdx %= (2*boxesPerSide - 1);
		return (faceIdx < boxesPerSide-1)? XNORMAL : YNORMAL;
	}


	/* Utility routines for getting neighbors */
	int getFaceIdxFromCoords_(int row, int col, int sli, int ell, FaceType type) {
		int boxesPerSide = 1 << ell;
		int facesPerSlice = 2 * (boxesPerSide-1)*boxesPerSide + boxesPerSide * boxesPerSide;
		if (type == XNORMAL) {
			int xnrmIdx = sli * facesPerSlice + row * (2*boxesPerSide - 1) + col;
			return xnrmIdx;
		} else if (type == YNORMAL) {
			int ynrmIdx = sli * facesPerSlice + row * (2*boxesPerSide - 1) + col + boxesPerSide-1;
			return ynrmIdx;
		} else if (type == ZNORMAL) {
			// IS THIS RIGHT?
			//int znrmIdx = sli * facesPerSlice + row * (2*boxesPerSide - 1) + col + 2*(boxesPerSide-1)*boxesPerSide;
			int znrmIdx = sli * facesPerSlice + row * boxesPerSide + col + 2*(boxesPerSide-1)*boxesPerSide;
			return znrmIdx;
		} else {
			throw runtime_error("Unrecognized face type!");
		}
	}

	// Note: returning a tensoridx3d here is kind of a hack
	TensorIdx3D getFaceCoordsFromIdx_(int idx, int ell) {
		int boxesPerSide = 1 << ell;
		int facesPerSlice = 2 * (boxesPerSide-1)*boxesPerSide + boxesPerSide * boxesPerSide;
		FaceType type = getFaceType_(idx, ell);
		if (type == XNORMAL) {
			// Do nothing
			;
		} else if(type == YNORMAL) {
			idx -= boxesPerSide-1;
		} else if(type == ZNORMAL) {
			idx -= 2*(boxesPerSide-1)*boxesPerSide;
			int z = idx / facesPerSlice;
			idx -= z * facesPerSlice;
			int y = idx / boxesPerSide;
			idx -= y * boxesPerSide;
			int x = idx;
			return TensorIdx3D(x,y,z,0,0,0);
		} else {
			throw runtime_error("Unrecognized face type!");
		}

		int z = idx / facesPerSlice;
		idx -= z * facesPerSlice;
		int y = idx / (2*boxesPerSide - 1);
		idx -= y * (2*boxesPerSide - 1);
		int x = idx;

		return TensorIdx3D(x,y,z,0,0,0);
	}

	IdxVec getSharedFaceChildrenOfBox_(int boxIdx, int ell) {
		set<int> chldFaces;
		IdxVec chldBoxes = getChildIdxFromBoxIdx_(boxIdx, ell);
		for (vector<int>::iterator it = chldBoxes.begin(); it != chldBoxes.end(); it++) {
			int chldIdx = *it;
			IdxVec faces = getNborFacesOfBox_(chldIdx, ell+1);
			for (vector<int>::iterator faceIt = faces.begin(); faceIt!= faces.end(); faceIt++) {
				int faceIdx = *faceIt;
				if (faceIdx == -1) {
					continue;
				}
				TensorIdx3D faceCoords = getFaceCoordsFromIdx_(faceIdx, ell+1);
				FaceType type = getFaceType_(faceIdx, ell+1);
				if ((type == XNORMAL && faceCoords.x() % 2 == 1) || (type == YNORMAL && faceCoords.y() % 2 == 1) || (type == ZNORMAL && faceCoords.z() % 2 == 1)) {
					// exterior face, include
					chldFaces.insert(faceIdx);
				}
			}
		}
		return IdxVec(chldFaces.begin(), chldFaces.end());
	}

	IdxVec getSelfFaceChildrenOfBox_(int boxIdx, int ell) {
		set<int> chldFaces;
		IdxVec chldBoxes = getChildIdxFromBoxIdx_(boxIdx, ell);
		for (vector<int>::iterator it = chldBoxes.begin(); it != chldBoxes.end(); it++) {
			int chldIdx = *it;
			IdxVec faces = getNborFacesOfBox_(chldIdx, ell+1);
			for (vector<int>::iterator faceIt = faces.begin(); faceIt!= faces.end(); faceIt++) {
				int faceIdx = *faceIt;
				if (faceIdx == -1) {
					continue;
				}
				TensorIdx3D faceCoords = getFaceCoordsFromIdx_(faceIdx, ell+1);
				FaceType type = getFaceType_(faceIdx, ell+1);
				if ((type == XNORMAL && faceCoords.x() % 2 == 0) || (type == YNORMAL && faceCoords.y() % 2 == 0) || (type == ZNORMAL && faceCoords.z() % 2 == 0)) {
					// interior face, include
					chldFaces.insert(faceIdx);
				}
			}
		}
		return IdxVec(chldFaces.begin(), chldFaces.end());
	}



		/* Utility routines for getting parents and children */
	int getParentIdxFromBoxIdx_(int boxIdx, int ell){
		int boxesPerSide = 1 << ell;
		TensorIdx3D boxCoord(boxIdx, boxesPerSide, boxesPerSide, boxesPerSide);
		TensorIdx3D parentCoord(boxCoord.x() / 2, boxCoord.y() / 2, boxCoord.z() / 2, boxesPerSide/2, boxesPerSide / 2, boxesPerSide / 2);
		return parentCoord.LinearIndex();
	}

	IdxVec getChildIdxFromFaceIdx_(int idx, int ell) {

		FaceType type = getFaceType_(idx,ell);
		IdxVec childInds;

		TensorIdx3D coord = getFaceCoordsFromIdx_(idx, ell);

		int row   = coord.y();
		int col   = coord.x();
		int slice = coord.z();

		// Everything has four children
		int childIdx1, childIdx2, childIdx3, childIdx4;
		if (type == XNORMAL) {
			childIdx1 = getFaceIdxFromCoords_(2 * row, 2 * col + 1, 2 * slice, ell + 1, XNORMAL);
			childIdx2 = getFaceIdxFromCoords_(2 * row, 2 * col + 1, 2 * slice + 1, ell + 1, XNORMAL);
			childIdx3 = getFaceIdxFromCoords_(2 * row + 1, 2 * col + 1, 2 * slice, ell + 1, XNORMAL);
			childIdx4 = getFaceIdxFromCoords_(2 * row + 1, 2 * col + 1, 2 * slice + 1, ell + 1, XNORMAL);
		} else if (type == YNORMAL) {
			childIdx1 = getFaceIdxFromCoords_(2 * row + 1, 2 * col, 2 * slice, ell + 1, YNORMAL);
			childIdx2 = getFaceIdxFromCoords_(2 * row + 1, 2 * col, 2 * slice + 1, ell + 1, YNORMAL);
			childIdx3 = getFaceIdxFromCoords_(2 * row + 1, 2 * col + 1, 2 * slice, ell + 1, YNORMAL);
			childIdx4 = getFaceIdxFromCoords_(2 * row + 1, 2 * col + 1, 2 * slice + 1, ell + 1, YNORMAL);
		} else if (type == ZNORMAL) {
			childIdx1 = getFaceIdxFromCoords_(2 * row, 2 * col, 2 * slice + 1, ell + 1, ZNORMAL);
			childIdx2 = getFaceIdxFromCoords_(2 * row + 1, 2 * col, 2 * slice + 1, ell + 1, ZNORMAL);
			childIdx3 = getFaceIdxFromCoords_(2 * row, 2 * col + 1, 2 * slice + 1, ell + 1, ZNORMAL);
			childIdx4 = getFaceIdxFromCoords_(2 * row + 1, 2 * col + 1, 2 * slice + 1, ell + 1, ZNORMAL);
		} else {
			throw runtime_error("Unrecognized face type!");
		}
		childInds.push_back(childIdx1);
		childInds.push_back(childIdx2);
		childInds.push_back(childIdx3);
		childInds.push_back(childIdx4);

		return childInds;
	}

	IdxVec getChildIdxFromBoxIdx_(int idx, int ell) {
		IdxVec childInds;
		int boxesPerSide = 1 << ell;
		TensorIdx3D coord(idx, boxesPerSide, boxesPerSide, boxesPerSide);

		childInds.push_back(TensorIdx3D(2*coord.x(), 2*coord.y(), 2*coord.z(), 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x()+1, 2*coord.y(), 2*coord.z(), 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x(), 2*coord.y()+1, 2*coord.z(), 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x(), 2*coord.y(), 2*coord.z()+1, 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x()+1, 2*coord.y()+1, 2*coord.z(), 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x()+1, 2*coord.y(), 2*coord.z()+1, 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x(), 2*coord.y()+1, 2*coord.z()+1, 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());
		childInds.push_back(TensorIdx3D(2*coord.x()+1, 2*coord.y()+1, 2*coord.z()+1, 2*boxesPerSide, 2*boxesPerSide, 2*boxesPerSide).LinearIndex());

		return childInds;
	}


	/* Given the index of a box, find the indices of the faces bordering it  */
	IdxVec getNborFacesOfBox_(int boxIdx, int ell) {
		int boxesPerSide = 1 << ell;
		TensorIdx3D boxCoord(boxIdx, boxesPerSide, boxesPerSide, boxesPerSide);

		IdxVec faceIdxs;

		// First, the x-normal faces
		int xNormRow = boxCoord.y(), xNormSlice = boxCoord.z();
		if (boxCoord.x() > 0) {
			// This box has a left face
			int xNormColumn = boxCoord.x() - 1;
			int xNormIdx = getFaceIdxFromCoords_(xNormRow, xNormColumn, xNormSlice, ell, XNORMAL);
			faceIdxs.push_back(xNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(xNormIdx,ell) == XNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}
		if (boxCoord.x() < boxesPerSide - 1) {
			// This box has a right face
			int xNormColumn = boxCoord.x();
			int xNormIdx = getFaceIdxFromCoords_(xNormRow, xNormColumn, xNormSlice, ell, XNORMAL);
			faceIdxs.push_back(xNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(xNormIdx,ell) == XNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}

		// Now the y-normal faces
		int yNormColumn = boxCoord.x(), yNormSlice = boxCoord.z();

		if (boxCoord.y() > 0) {
			// This box has a bottom face
			int yNormRow = boxCoord.y() - 1;
			int yNormIdx = getFaceIdxFromCoords_(yNormRow, yNormColumn, yNormSlice, ell, YNORMAL);
			faceIdxs.push_back(yNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(yNormIdx,ell) == YNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}
		if (boxCoord.y() < boxesPerSide - 1) {
			// This box has a top face
			int yNormRow = boxCoord.y();
			int yNormIdx = getFaceIdxFromCoords_(yNormRow, yNormColumn, yNormSlice, ell, YNORMAL);
			faceIdxs.push_back(yNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(yNormIdx,ell) == YNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}

		// Now the z-normal faces
		int zNormRow = boxCoord.y(), zNormColumn = boxCoord.x();

		if (boxCoord.z() > 0) {
			// This box has a below face
			int zNormSlice = boxCoord.z() - 1;
			int zNormIdx   = getFaceIdxFromCoords_(zNormRow, zNormColumn, zNormSlice, ell, ZNORMAL);
			faceIdxs.push_back(zNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(zNormIdx,ell) == ZNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}
		if (boxCoord.z() < boxesPerSide - 1) {
			// This box has an above face
			int zNormSlice = boxCoord.z();
			int zNormIdx   = getFaceIdxFromCoords_(zNormRow, zNormColumn, zNormSlice, ell, ZNORMAL);
			faceIdxs.push_back(zNormIdx);
			#ifdef DEBUG
			assert(getFaceType_(zNormIdx,ell) == ZNORMAL);
			#endif
		}
		else {
			faceIdxs.push_back(-1);
		}

		return faceIdxs;
	}

	IdxVec getNborBoxesOfFace_(int idx, int ell) {
		IdxVec nbors;
		int boxesPerSide = 1 << ell;

		TensorIdx3D coord = getFaceCoordsFromIdx_(idx,ell);
		int col   = coord.x();
		int row   = coord.y();
		int slice = coord.z();
		FaceType type = getFaceType_(idx,ell);

		if (type == XNORMAL) {
			// Face only has left and right neighbors

			TensorIdx3D boxCoordL(col, row, slice, boxesPerSide, boxesPerSide, boxesPerSide);
			TensorIdx3D boxCoordR(col+1, row, slice, boxesPerSide, boxesPerSide, boxesPerSide);
			nbors.push_back(boxCoordL.LinearIndex());
			nbors.push_back(boxCoordR.LinearIndex());
			return nbors;
		} else if (type == YNORMAL) {
			// Face only has up and down neighbors
			TensorIdx3D boxCoordB(col, row, slice, boxesPerSide, boxesPerSide, boxesPerSide);
			TensorIdx3D boxCoordT(col, row+1, slice, boxesPerSide, boxesPerSide, boxesPerSide);
			nbors.push_back(boxCoordB.LinearIndex());
			nbors.push_back(boxCoordT.LinearIndex());
			return nbors;
		} else if (type == ZNORMAL) {
			// Face only has above and below neighbors
			TensorIdx3D boxCoordB(col, row, slice, boxesPerSide, boxesPerSide, boxesPerSide);
			TensorIdx3D boxCoordA(col, row, slice+1, boxesPerSide, boxesPerSide, boxesPerSide);
			nbors.push_back(boxCoordB.LinearIndex());
			nbors.push_back(boxCoordA.LinearIndex());
			return nbors;
		} else {
			throw runtime_error("Unrecognized face type!");
		}
	}


	IdxVec getNborFacesOfFace_(int idx, int ell) {
		IdxVec nborBoxes = getNborBoxesOfFace_(idx, ell);
		IdxVec nbors;
		nbors.reserve(10);
		for (int i = 0; i < nborBoxes.size(); i++) {
			IdxVec nborFaces = getNborFacesOfBox_(nborBoxes[i], ell);
			for (int j = 0; j < nborFaces.size(); j++) {
				if (nborFaces[j] != idx) {
					nbors.push_back(nborFaces[j]);
				}
			}
		}
		return nbors;
	}




}; // namespace TreeFactor3D