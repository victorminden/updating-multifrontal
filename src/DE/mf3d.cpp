#include "mf.hpp"

namespace TreeFactor3D {
	using namespace LinAlg;
	using namespace std;


	template <class Scalar>
	void MF3D<Scalar>::help(){
		for (int ell = levels_-1; ell >=0; ell--){
			int boxesPerSide = 1 << ell;
			int totalBoxes = boxesPerSide * boxesPerSide * boxesPerSide;
			for (int i = 0; i < totalBoxes; i++) {
				IdxVec slf = boxData_[ell][i].bdDOFs;
				cout << i << endl;
				for (int j = 0; j < slf.size(); j++){
					TensorIdx3D temp(slf[j],Nx_, Ny_, Nz_);
					cout << temp.x() << ' ' << temp.y() << ' ' << temp.z() << ' ';
				}
				cout << endl;
			}
		}
	}

		/* Primary construtor */
	template <class Scalar>
	MF3D<Scalar>::MF3D(const LinAlg::Sparse<Scalar>& A, int Nx, int Ny, int Nz, int width, int maxSideLength)
		: A_(A), Nx_(Nx), Ny_(Ny), Nz_(Nz), width_(width) {
		#ifdef VERBOSE
			cout << LINE << endl;
			cout << "Entering MF3D constructor..." << endl;
		#endif
		#ifdef HASCPP11TIMING
			durationLA_ = 0;
		  	durationData_ = 0;
		#endif
		/*
			First, compute the tree depth.  We enforce that the sides of MOST boxes at the lowest
			level have a length of less than maxSideLength
		*/
		int topSideLength = Nx;
		topSideLength = (topSideLength < Ny)? Ny : topSideLength;
		topSideLength = (topSideLength < Nz)? Nz : topSideLength;
		/*
			The side length at the top is 2^levels * the side length at the bottom
		 	plus (2^levels-1) * the separator width
		 	this implies 2^levels * (maxSideLength + width) = topSideLength + width
		 */
		double num = topSideLength + width;
		double den = maxSideLength + width;
		levels_ = (int) ceil(log2(num/den));

		#ifdef VERBOSE
			cout << "Computed recursion depth is " << levels_ << " levels." << endl;
		#endif


		if(levels_ <= 1) {
			// We have to recurse.  If we don't recurse, we're not doing multifrontal.  Throw an error.
			// TODO: This occasionally doesn't like input that it should because of the unequal spacing and floor
			// in computing levels_
			throw runtime_error("Based on the input parameters, the recursion depth is 1 level (i.e., do not recurse).  Try increasing the problem size or decreasing maxSideLength.");
		}

		// Get the space to set up the first level
		boxData_.resize(levels_);
		faceData_.resize(levels_ - 1);


		int boxesPerSide = 1 << levels_ - 1;
		int totalBoxes = boxesPerSide * boxesPerSide * boxesPerSide;
		int totalFaces = 3 * boxesPerSide * boxesPerSide * (boxesPerSide - 1);
		boxData_[levels_-1].resize(totalBoxes);
		faceData_[levels_-2].resize(totalFaces);

		// If the number of boxes doesn't make for even division of points, put more in the last one
		int boxWidthX = (int) floor((double)(Nx - (boxesPerSide - 1) * width) / (double)(boxesPerSide) );
		int boxWidthY = (int) floor((double)(Ny - (boxesPerSide - 1) * width) / (double)(boxesPerSide) );
		int boxWidthZ = (int) floor((double)(Nz - (boxesPerSide - 1) * width) / (double)(boxesPerSide) );

		// Sanity checks
		bool tooManyXPoints = (Nx_ < boxWidthX * (boxesPerSide - 1) + width*(boxesPerSide-1));
		bool tooManyYPoints = (Ny_ < boxWidthY * (boxesPerSide - 1) + width*(boxesPerSide-1));
		bool tooManyZPoints = (Nz_ < boxWidthZ * (boxesPerSide - 1) + width*(boxesPerSide-1));
		if (tooManyXPoints || tooManyYPoints || tooManyZPoints) {
			throw runtime_error("You found a bug!  The math for computing the number of points per box at a leaf level broke on this input.");
		}


		boxWidthXBottom_ = boxWidthX;
		boxWidthYBottom_ = boxWidthY;
		boxWidthZBottom_ = boxWidthZ;


		/* Add the bottom level of boxes to the marked set and set up DOFs */
		for (int boxIdx = 0; boxIdx < totalBoxes; boxIdx++) {
			markedBoxes_.insert(boxIdx);
			Box& crntBox = boxData_[levels_-1][boxIdx];

			// Find coordinate of this box
			TensorIdx3D boxCoord(boxIdx, boxesPerSide, boxesPerSide, boxesPerSide);
			// Find corresponding lower-left DOF
			int firstDOFX = boxCoord.x() * (boxWidthX + width);
			int firstDOFY = boxCoord.y() * (boxWidthY + width);
			int firstDOFZ = boxCoord.z() * (boxWidthZ + width);

			int xUB = (boxCoord.x() == boxesPerSide-1)? Nx : firstDOFX + boxWidthX;
			int yUB = (boxCoord.y() == boxesPerSide-1)? Ny : firstDOFY + boxWidthY;
			int zUB = (boxCoord.z() == boxesPerSide-1)? Nz : firstDOFZ + boxWidthZ;

			IdxVec& self = crntBox.intDOFs;
			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx + k * Nx * Ny;
						self.push_back(idx);
					}
				}
			}
			// Also populate faces (know DOF set)
			// Every face seen twice so we only do our left/top/below (0,3,4)
			IdxVec faces = getNborFacesOfBox_(boxIdx, levels_ - 1);
			if (boxCoord.x() > 0) {
				// have left face
				IdxVec& leftDOFs = faceData_[levels_-2][faces[0]].DOFs;

				int firstFaceDOFX = firstDOFX - width;
				int firstFaceDOFY = firstDOFY;
				int firstFaceDOFZ = firstDOFZ;

				int xFaceUB = firstFaceDOFX + width;
				int yFaceUB = (boxCoord.y() == boxesPerSide-1)? Ny : firstFaceDOFY + boxWidthY;
				int zFaceUB = (boxCoord.z() == boxesPerSide-1)? Nz : firstFaceDOFZ + boxWidthZ;

				for (int k =firstFaceDOFZ; k < zFaceUB; k++){
					for (int j = firstFaceDOFY; j < yFaceUB; j++) {
						for (int i = firstFaceDOFX; i < xFaceUB; i++) {
							leftDOFs.push_back(i + j * Nx + k * Nx * Ny);
						}
					}
				}
			}


			if (boxCoord.y() < boxesPerSide-1) {
				// have top face
				IdxVec& topDOFs = faceData_[levels_-2][faces[3]].DOFs;

				int firstFaceDOFX = firstDOFX;
				int firstFaceDOFY = yUB;
				int firstFaceDOFZ = firstDOFZ;

				int xFaceUB = (boxCoord.x() == boxesPerSide-1)? Nx : firstFaceDOFX + boxWidthX;
				int yFaceUB = firstFaceDOFY + width;
				int zFaceUB = (boxCoord.z() == boxesPerSide-1)? Nz : firstFaceDOFZ + boxWidthZ;

				for (int k =firstFaceDOFZ; k < zFaceUB; k++){
					for (int j = firstFaceDOFY; j < yFaceUB; j++) {
						for (int i = firstFaceDOFX; i < xFaceUB; i++) {
							topDOFs.push_back(i + j * Nx + k * Nx * Ny);
						}
					}
				}
			}

			if (boxCoord.z() > 0) {
				// have below face
				IdxVec& belowDOFs = faceData_[levels_-2][faces[4]].DOFs;

				int firstFaceDOFX = firstDOFX;
				int firstFaceDOFY = firstDOFY;
				int firstFaceDOFZ = firstDOFZ - width;

				int xFaceUB = (boxCoord.x() == boxesPerSide-1)? Nx : firstFaceDOFX + boxWidthX;
				int yFaceUB = (boxCoord.y() == boxesPerSide-1)? Ny : firstFaceDOFY + boxWidthY;
				int zFaceUB = firstFaceDOFZ + width;

				for (int k =firstFaceDOFZ; k < zFaceUB; k++){
					for (int j = firstFaceDOFY; j < yFaceUB; j++) {
						for (int i = firstFaceDOFX; i < xFaceUB; i++) {
							belowDOFs.push_back(i + j * Nx + k * Nx * Ny);
						}
					}
				}
			}


		}

		for (int ell = levels_ - 2; ell >= 0; ell--) {
			int boxesPerSide = 1 << ell;
			if (ell != 0) {
				int totalFaces = 3*boxesPerSide*boxesPerSide*(boxesPerSide-1);
				faceData_[ell-1].resize(totalFaces);
			}

			/* Ask for space */
			boxData_[ell].resize(boxesPerSide * boxesPerSide * boxesPerSide);
		}


		#ifdef VERBOSE
			cout << "Leaving MF3D constructor." << endl;
		#endif

	}

	template <class Scalar>
	void MF3D<Scalar>::Update(const Sparse<Scalar>& A, set<int> modifiedDOFs) {
		markedBoxes_.clear();
		markedFaces_.clear();
		#ifdef HASCPP11TIMING
			durationLA_ = 0;
			durationData_ = 0;
		#endif
		A_ = A;

		/* TODO: fill in */
		int ell = levels_ - 1;
		int boxesPerSide = 1 << ell;


		for (set<int>::iterator it = modifiedDOFs.begin(); it != modifiedDOFs.end(); it++) {
			// Find box that corresponds to this DOF
			int idx = *it;
			int i = idx % Nx_;
			idx /= Nx_;
			int j = idx % Ny_;
			idx /= Ny_;
			int k = idx;

			//cout << i << ' ' << j << endl;
			int iBox = i / (boxWidthXBottom_ + width_);
			int jBox = j / (boxWidthYBottom_ + width_);
			int kBox = k / (boxWidthZBottom_ + width_);
			TensorIdx3D boxCoord(iBox, jBox, kBox, boxesPerSide, boxesPerSide, boxesPerSide);
			// cout << iBox << ' ' << jBox << ' ' << kBox << ' ' << endl;
			// cout << boxCoord.LinearIndex() << endl;
			markedBoxes_.insert(boxCoord.LinearIndex());
			//cout << iBox << ' ' << jBox << endl;
		}

		// int totalBoxes = boxesPerSide * boxesPerSide;
		// for (int i = 0; i < totalBoxes; i++){
		// 	markedBoxes_.insert(i);
		// }
		Factor();
	}


	/* Utility functions for getting DOFs */

	template <class Scalar>
	IdxVec MF3D<Scalar>::getBoxCenterPoints_(int boxIdx, int ell) {
		/* At the middle of a box that's not a leaf, there is a set of points at the center of a cross
		   that must be added to the active point set for the box.  This routine finds those DOFS.*/
		int boxesPerSide = 1 << ell;
		IdxVec dofs;
		set<int> dofSet;
		TensorIdx3D coord(boxIdx, boxesPerSide, boxesPerSide, boxesPerSide);

		int distFromBottom = levels_-1 - ell;

		int boxWidthX = boxWidthXBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthY = boxWidthYBottom_ * (1 << distFromBottom) + ((1 << distFromBottom )- 1) * width_;
		int boxWidthZ = boxWidthZBottom_ * (1 << distFromBottom) + ((1 << distFromBottom )- 1) * width_;
		distFromBottom--;
		int boxWidthXBelow = boxWidthXBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthYBelow = boxWidthYBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthZBelow = boxWidthZBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;

		int row = coord.y();
		int col = coord.x();
		int sli = coord.z();

		// x-aligned line

		int firstDOFX = col * (boxWidthX + width_);
		int firstDOFY = row * (boxWidthY + width_) + boxWidthYBelow;
		int firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZBelow;

		int xUB = firstDOFX + boxWidthX;
		int yUB = firstDOFY + width_;
		int zUB = firstDOFZ + width_;
		if (col == boxesPerSide - 1) {
			xUB = Nx_;
		}
		for (int k = firstDOFZ; k < zUB; k++) {
			for (int j = firstDOFY; j < yUB; j++) {
				for (int i = firstDOFX; i < xUB; i++) {
					int idx = i + j * Nx_ + k * Nx_ * Ny_;
					dofSet.insert(idx);
				}
			}
		}


		// y-aligned line

		firstDOFX = col * (boxWidthX + width_) + boxWidthXBelow;
		firstDOFY = row * (boxWidthY + width_);
		firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZBelow;

		xUB = firstDOFX + width_;
		yUB = firstDOFY + boxWidthY;
		zUB = firstDOFZ + width_;
		if (row == boxesPerSide - 1) {
			yUB = Ny_;
		}
		for (int k = firstDOFZ; k < zUB; k++) {
			for (int j = firstDOFY; j < yUB; j++) {
				for (int i = firstDOFX; i < xUB; i++) {
					int idx = i + j * Nx_ + k * Nx_ * Ny_;
					dofSet.insert(idx);
				}
			}
		}


		// z-aligned line

		firstDOFX = col * (boxWidthX + width_) + boxWidthXBelow;
		firstDOFY = row * (boxWidthY + width_) + boxWidthYBelow;
		firstDOFZ = sli * (boxWidthZ + width_);

		xUB = firstDOFX + width_;
		yUB = firstDOFY + width_;
		zUB = firstDOFZ + boxWidthZ;
		if (sli == boxesPerSide - 1) {
			zUB = Nz_;
		}
		for (int k = firstDOFZ; k < zUB; k++) {
			for (int j = firstDOFY; j < yUB; j++) {
				for (int i = firstDOFX; i < xUB; i++) {
					int idx = i + j * Nx_ + k * Nx_ * Ny_;
					dofSet.insert(idx);
				}
			}
		}
		dofs.insert(dofs.end(), dofSet.begin(), dofSet.end());

		// for (int i = 0; i < dofs.size(); i++) {
		// 	TensorIdx3D a(dofs[i], Nx_, Ny_, Nz_);
		// 	cout << a.x() << ' ' << a.y() << ' ' << a.z() << ' ';
		// }
		// exit(0);
		return dofs;
	}

	template <class Scalar>
	IdxVec MF3D<Scalar>::getFaceCenterPoints_(int faceIdx, int ell) {
	/* At the middle of a parent face, there is a set of points between the child faces
	   that must be added to the active point set for the parent face.  This routine finds those DOFS.*/
		IdxVec dofs;
		set<int> dofSet;
		TensorIdx3D coord = getFaceCoordsFromIdx_(faceIdx, ell);
		FaceType type = getFaceType_(faceIdx, ell);
		int row = coord.y();
		int col = coord.x();
		int sli = coord.z();

		int boxesPerSide = 1 << ell;
		int distFromBottom = levels_-1 - ell;

		int boxWidthX = boxWidthXBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthY = boxWidthYBottom_ * (1 << distFromBottom) + ((1 << distFromBottom )- 1) * width_;
		int boxWidthZ = boxWidthZBottom_ * (1 << distFromBottom) + ((1 << distFromBottom )- 1) * width_;

		distFromBottom--;

		int boxWidthXBelow = boxWidthXBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthYBelow = boxWidthYBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;
		int boxWidthZBelow = boxWidthZBottom_ * (1 << distFromBottom) + ((1 << distFromBottom) - 1) * width_;

		if (type == YNORMAL) {

			// First the line along the z-axis
			int firstDOFX = col * (boxWidthX + width_) + boxWidthXBelow;
			int firstDOFY = row * (boxWidthY + width_) + boxWidthY;
			int firstDOFZ = sli * (boxWidthZ + width_);

			int xUB = firstDOFX + width_;
			int yUB = firstDOFY + width_;
			int zUB = firstDOFZ + boxWidthZ;
			if (sli == boxesPerSide - 1) {
				zUB = Nz_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}
			// Now the line along the x-axis
			firstDOFX = col * (boxWidthX + width_);
			firstDOFY = row * (boxWidthY + width_) + boxWidthY;
			firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZBelow;

			xUB = firstDOFX + boxWidthX;
			yUB = firstDOFY + width_;
			zUB = firstDOFZ + width_;

			if (col == boxesPerSide - 1) {
				xUB = Nx_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}

		} else if (type == XNORMAL) {

			// First the line along the z-axis
			int firstDOFX = col * (boxWidthX + width_) + boxWidthX;
			int firstDOFY = row * (boxWidthY + width_) + boxWidthYBelow;
			int firstDOFZ = sli * (boxWidthZ + width_);

			int xUB = firstDOFX + width_;
			int yUB = firstDOFY + width_;
			int zUB = firstDOFZ + boxWidthZ;
			if (sli == boxesPerSide - 1) {
				zUB = Nz_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}


			// Now the line along the y-axis
			firstDOFX = col * (boxWidthX + width_) + boxWidthX;
			firstDOFY = row * (boxWidthY + width_);
			// Think about this
			firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZBelow;

			xUB = firstDOFX + width_;
			yUB = firstDOFY + boxWidthY;
			zUB = firstDOFZ + width_;
			if (row == boxesPerSide - 1) {
				yUB = Ny_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}

		} else if (type == ZNORMAL) {

			// First the line along the x-axis
			int firstDOFX = col * (boxWidthX + width_);
			int firstDOFY = row * (boxWidthY + width_) + boxWidthYBelow;
			int firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZ;

			int xUB = firstDOFX + boxWidthX;
			int yUB = firstDOFY + width_;
			int zUB = firstDOFZ + width_;
			if (col == boxesPerSide - 1) {
				xUB = Nx_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}


			// Now the line along the y-axis
			firstDOFX = col * (boxWidthX + width_) + boxWidthXBelow;
			firstDOFY = row * (boxWidthY + width_);
			firstDOFZ = sli * (boxWidthZ + width_) + boxWidthZ;

			xUB = firstDOFX + width_;
			yUB = firstDOFY + boxWidthY;
			zUB = firstDOFZ + width_;

			if (row == boxesPerSide - 1) {
				yUB = Ny_;
			}

			for (int k = firstDOFZ; k < zUB; k++) {
				for (int j = firstDOFY; j < yUB; j++) {
					for (int i = firstDOFX; i < xUB; i++) {
						int idx = i + j * Nx_ + k * Nx_ * Ny_;
						dofSet.insert(idx);
					}
				}
			}

		}
		// cout << "TYPE is " << type;
		dofs.insert(dofs.end(),dofSet.begin(),dofSet.end());
		// for (int i = 0; i < dofs.size(); i++){
		// 	int x =TensorIdx3D(dofs[i],Nx_,Ny_,Nz_).x();
		// 	int y =TensorIdx3D(dofs[i],Nx_,Ny_,Nz_).y();
		// 	int z =TensorIdx3D(dofs[i],Nx_,Ny_,Nz_).z();
		// 	cout << x << ' ' << y << ' ' << z << ' ';
		// }
		// cout << endl << endl;
		// exit(0);
		return dofs;

	}


	/* Update the data for a box/face as we step up a level */
	template <class Scalar>
	void MF3D<Scalar>::updateFaceData_(int ell) {
		#ifdef VERBOSE
			cout << "Updating face data for level " << ell << "..." << endl;
		#endif
		// Loop over each child face and pull up its data, then add center point data
		for (set<int>::iterator it = markedFaces_.begin(); it != markedFaces_.end(); it++) {
			int faceIdx = *it;
			if (faceIdx == -1) {
				continue;
			}
			Dense<Scalar>& A11  = faceData_[ell-1][faceIdx].A11;
			IdxVec&        self = faceData_[ell-1][faceIdx].DOFs;

    		A11 = Dense<Scalar>();
    		self = IdxVec();


			// Each child face has DOFs that now belong to me, add those to my active DOF set
			IdxVec childIdxs = getChildIdxFromFaceIdx_(faceIdx, ell);
			for (vector<int>::iterator childIt = childIdxs.begin(); childIt != childIdxs.end(); childIt++) {
				int childIdx = *childIt;
				IdxVec& chld = faceData_[ell][childIdx].DOFs;
				self.insert(self.end(), chld.begin(), chld.end());
			}
			// cout << "Size is: " << self.size() << endl;

			// for (int i = 0; i < self.size(); i++){
			// 	int x =TensorIdx3D(self[i],Nx_,Ny_,Nz_).x();
			// 	int y =TensorIdx3D(self[i],Nx_,Ny_,Nz_).y();
			// 	int z =TensorIdx3D(self[i],Nx_,Ny_,Nz_).z();
			// 	cout << x << ' ' << y << ' ' << z << ' ';
			// }
			// cout << endl;
		// cout << "CENTER POINTS" << endl;
			// Get the center points between my child faces and add to my active DOF set
			IdxVec centerPoints = getFaceCenterPoints_(faceIdx, ell);
			self.insert(self.end(), centerPoints.begin(), centerPoints.end());

			A11.resize(self.size(),self.size());

			// The center points have not had any Schur updates, so grab their interactions out of the sparse matrix
			AddNewInteractions_(A11, self, centerPoints);

			// Now that A11 is big enough, pull up Schur-ed interactions from child faces
			for (vector<int>::iterator childIt = childIdxs.begin(); childIt != childIdxs.end(); childIt++) {
				int chldIdx      = *childIt;
				IdxVec& faceDOFs   = faceData_[ell][chldIdx].DOFs;
				Dense<Scalar>& A11Other = faceData_[ell][chldIdx].A11;
				ScatterReplace_(A11, self, A11Other, faceDOFs);
			}

		}
		markedFaces_.clear();
	}


	template <class Scalar>
	void MF3D<Scalar>::updateBoxData_(int ell) {
		#ifdef VERBOSE
			cout << "Updating box data for level " << ell << "..." << endl;
		#endif
		/* Loop over each marked box and pull up updated data from children */
		for (set<int>::iterator it = markedBoxes_.begin(); it != markedBoxes_.end(); it++) {
			int boxIdx = *it;

			Dense<Scalar>& A22   = boxData_[ell][boxIdx].A22;
			Dense<Scalar>& A11   = boxData_[ell][boxIdx].A11post;
			IdxVec&        self  = boxData_[ell][boxIdx].intDOFs;
			IdxVec&        nbors = boxData_[ell][boxIdx].bdDOFs;

			Dense<Scalar>& A12   = boxData_[ell][boxIdx].XL;
    		Dense<Scalar>& A21   = boxData_[ell][boxIdx].XR;

    		A22 = Dense<Scalar>();
    		A11 = Dense<Scalar>();
    		self = IdxVec();
    		nbors = IdxVec();
    		A12 = Dense<Scalar>();
    		A21 = Dense<Scalar>();



			// There are faces from the lower level that are interior to this box, pull up their DOFs
			IdxVec childFaces = getSelfFaceChildrenOfBox_(boxIdx, ell);
			for (vector<int>::iterator childIt = childFaces.begin(); childIt != childFaces.end(); childIt++) {
				int childFaceIdx = *childIt;
				IdxVec& chld = faceData_[ell][childFaceIdx].DOFs;
				self.insert(self.end(), chld.begin(), chld.end());
			}


			// The child faces border a small set of center points, add that set of DOFs to me too
			IdxVec centerPoints = getBoxCenterPoints_(boxIdx, ell);
			self.insert(self.end(), centerPoints.begin(), centerPoints.end());

			A22.resize(self.size(), self.size());
			// The center points have no Schur updates, grab their interactions from sparse matrix
			AddNewInteractions_(A22, self, centerPoints);



			// Talk to the big faces that border this box and figure out what DOFs will get Schur updates
			// at this level
			IdxVec nborFaces = getNborFacesOfBox_(boxIdx, ell);
			for (vector<int>::iterator nborIt = nborFaces.begin(); nborIt != nborFaces.end(); nborIt++) {
				int nborfaceIdx = *nborIt;
				if (nborfaceIdx == -1){
					continue;
				}
				IdxVec& nborDOFs = faceData_[ell-1][nborfaceIdx].DOFs;
				nbors.insert(nbors.end(), nborDOFs.begin(), nborDOFs.end());
			}


			// Now that we know how big A11 needs to be, resize it
			A11.resize(nbors.size(), nbors.size());
			A12.resize(nbors.size(), self.size());
			A21.resize(self.size(), nbors.size());


			// The new center points for neighboring faces have pure (not Schur-ed) interactions to me, grab those from sparse matrix
			for (vector<int>::iterator nborIt = nborFaces.begin(); nborIt != nborFaces.end(); nborIt++) {
				int nborfaceIdx = *nborIt;
				if (nborfaceIdx == -1){
					continue;
				}
				IdxVec centerPoints = getFaceCenterPoints_(nborfaceIdx, ell);
				AddNewRowInteractions_(A12, nbors,self,centerPoints);
				AddNewColInteractions_(A21, self,nbors,centerPoints);
			}




			/* At the child box level, we did a Schur update that clique-ified some faces that are interior to this box
			   with some faces that are outside this box now.  Grab those interactions.
			   Also, some of those updates affected exterior-exterior and interior-interior interactions, grab those too */

			IdxVec boxChildren = getChildIdxFromBoxIdx_(boxIdx, ell);
			for (vector<int>::iterator childIt = boxChildren.begin(); childIt != boxChildren.end(); childIt++) {
				int childIdx = *childIt;
				IdxVec& child = boxData_[ell+1][childIdx].bdDOFs;
				Dense<Scalar>& A11chld = boxData_[ell+1][childIdx].A11post;
				// blocks of big faces to me from clique from previous level updates
				ScatterReplace_(A12, nbors, self, A11chld, child);
				ScatterReplace_(A21, self, nbors, A11chld, child);

				/* Tack on the off-diagonal that becomes on-diagonal */
				ScatterReplace_(A22, self, A11chld, child);
				ScatterReplace_(A11, nbors, A11chld, child);
			}

			/* This should be unnecessary for multifrontal but needed for HIF -- copy any modified updates
			   from face children, to account for extra compression */
			childFaces = getSelfFaceChildrenOfBox_(boxIdx, ell);
			for (vector<int>::iterator childIt = childFaces.begin(); childIt != childFaces.end(); childIt++) {
				int childFaceIdx = *childIt;
				IdxVec& chld = faceData_[ell][childFaceIdx].DOFs;
				Dense<Scalar>& A22chld = faceData_[ell][childFaceIdx].A11;
				ScatterReplace_(A22, self, A22chld, chld);
			}

			/* This should be unnecessary for multifrontal but needed for HIF -- copy any modified updates
			   from face neighbor children, to account for extra compression */
			childFaces = getSharedFaceChildrenOfBox_(boxIdx, ell);
			for (vector<int>::iterator childIt = childFaces.begin(); childIt != childFaces.end(); childIt++) {
				int childFaceIdx = *childIt;
				IdxVec& child = faceData_[ell][childFaceIdx].DOFs;
				Dense<Scalar>& A11chld = faceData_[ell][childFaceIdx].A11;
				ScatterReplace_(A11, nbors, A11chld, child);
			}

		}
	}


	template <class Scalar>
	void MF3D<Scalar>::Factor() {
		#ifdef VERBOSE
			cout << LINE << endl;
			cout << "Beginning MF3D factorization..." << endl;
		#endif
		/* Loop over the levels from bottom of tree to top */

		/* Start at the bottom, do the first step of elimination */
		FactorLeafLevel_();
		/* For each other level, we pull up info from below as opposed to from the matrix */

		for (int ell = levels_ - 2; ell >= 0; ell--) {
			if (ell != 0) {
			/* Pull big face data from little face data and add center points */
				// int totalFaces = 3*boxesPerSide*boxesPerSide*(boxesPerSide-1);
				// faceData_[ell-1].resize(totalFaces);

				#ifdef HASCPP11TIMING
				auto t1 = chrono::high_resolution_clock::now();
				#endif

				updateFaceData_(ell);

				#ifdef HASCPP11TIMING
				auto t2 = chrono::high_resolution_clock::now();
				durationData_ += chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
				#endif
			}



			#ifdef HASCPP11TIMING
			auto t1 = chrono::high_resolution_clock::now();
			#endif

			/* Ask for space */
			//boxData_[ell].resize(boxesPerSide * boxesPerSide * boxesPerSide);
			updateBoxData_(ell);


			#ifdef HASCPP11TIMING
			auto t2 = chrono::high_resolution_clock::now();
			durationData_ += chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
			#endif

			/* Factor boxes */
			FactorUpperLevel_(ell);

			//help();

		}
		#ifdef VERBOSE
			cout << LINE << endl;
			cout << "MF3D factorization complete." << endl;
			cout << LINE << endl;
		#endif
		#ifdef HASCPP11TIMING
			cout << "Time in linear algebra was " << (double) durationLA_ / 1000.0 << " seconds.\n";
			cout << "Time updating data was " << (double) durationData_ / 1000.0 << " seconds.\n";
		#endif
	}



	template <class Scalar>
	void MF3D<Scalar>::FactorLeafLevel_() {
		#ifdef VERBOSE
			cout << "Factoring leaf level..." << endl;
		#endif

		int ell = levels_-1;

		for (set<int>::iterator it = markedBoxes_.begin(); it != markedBoxes_.end(); it++) {
			int boxIdx = *it;

			Dense<Scalar>& A11     = boxData_[ell][boxIdx].A11post;
			Dense<Scalar>& A22     = boxData_[ell][boxIdx].A22;
			Dense<Scalar>& A22Inv  = boxData_[ell][boxIdx].A22Inv;
			Dense<Scalar>& XL      = boxData_[ell][boxIdx].XL;
			Dense<Scalar>& XR      = boxData_[ell][boxIdx].XR;
			Dense<Scalar>& S       = boxData_[ell][boxIdx].SchurComp;

			A22    = Dense<Scalar>();
			A22Inv = Dense<Scalar>();
			S      = Dense<Scalar>();

			IdxVec& self  = boxData_[ell][boxIdx].intDOFs;
			IdxVec& nbors = boxData_[ell][boxIdx].bdDOFs;
			nbors = IdxVec();


			IdxVec nborFaces = getNborFacesOfBox_(boxIdx, ell);

			// faceCtr is "face counter"
			for (int faceCtr = 0; faceCtr < nborFaces.size(); faceCtr++) {
				int faceIdx = nborFaces[faceCtr];
				if (faceIdx == -1){
					continue;
				}
				IdxVec& faceDOFs = faceData_[ell-1][faceIdx].DOFs;
				nbors.insert(nbors.end(), faceDOFs.begin(), faceDOFs.end());
			}

			// Form elimination blocks XL and XR and pass Schur update to faces (mark)

			A22     = A_(self,self);
			A22Inv  = A22;

			Dense<Scalar> A21 = A_(self,nbors);
			Dense<Scalar> A12 = A_(nbors,self);

			//inv
			// Inv(A22Inv);
			// XL  = A12 * A22Inv;
			// XR  = A22Inv * A21;

			// lu

			#ifdef HASCPP11TIMING
			auto t1 = chrono::high_resolution_clock::now();
			#endif
			LU(A22Inv);
			XL = A12;
			LUSolveT(A22Inv, XL);
			XR = A21;
			LUSolve(A22Inv, XR);


			S    = A12 * XR;
			S    *= -1.0;

			// Update the cliqueified interaction between skeletons
			A11 = A_(nbors,nbors);
			A11 += S;

			#ifdef HASCPP11TIMING
			auto t2 = chrono::high_resolution_clock::now();
			durationLA_ += chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
			#endif

			markedFaces_.insert(nborFaces.begin(), nborFaces.end());
		}
		// Finished the box level, push it to the face level
		markedBoxes_.clear();
		set<int> newMarkedFaces;
		// Loop over all faces at this level and populate with info
		for (set<int>::iterator it = markedFaces_.begin(); it != markedFaces_.end(); it++) {
			int faceIdx = *it;
			if (faceIdx == -1) {
				continue;
			}

			Dense<Scalar>& A11  = faceData_[ell-1][faceIdx].A11;
			IdxVec&        self = faceData_[ell-1][faceIdx].DOFs;

			// In upper levels, replace this with grabbing from children entirely
			A11 = A_(self,self);

			IdxVec nborBoxes = getNborBoxesOfFace_(faceIdx, ell);

			for (int boxCtr = 0; boxCtr < 2; boxCtr++){
				// grab my diagonal update
				int boxIdx    = nborBoxes[boxCtr];
				IdxVec& bdDOFs   = boxData_[ell][boxIdx].bdDOFs;
				Dense<Scalar>& S = boxData_[ell][boxIdx].SchurComp;

				ScatterAdd_(A11, self, S, bdDOFs);

				// Mark parent boxes of nbor
				int prntIdx = getParentIdxFromBoxIdx_(boxIdx,ell);
				markedBoxes_.insert(prntIdx);
				// TODO: verify Marking parents faces here
				if (ell > 1) {
					IdxVec faceNbrs = getNborFacesOfBox_(prntIdx, ell - 1);
					newMarkedFaces.insert(faceNbrs.begin(), faceNbrs.end());
				}
			}


			//A11.print();
		}
		markedFaces_ = newMarkedFaces;
		// TODO: mark parent faces?

		#ifdef VERBOSE
			cout << "Finished leaf level." << endl;
		#endif
	}

	template <class Scalar>
	void MF3D<Scalar>::FactorUpperLevel_(int ell) {
		#ifdef VERBOSE
			cout << "Factoring level " << ell << "..." << endl;
		#endif
		if (ell == 0){
			Dense<Scalar>& A22     = boxData_[ell][0].A22;
			Dense<Scalar>& A22Inv  = boxData_[ell][0].A22Inv;
			A22Inv = A22;

			// Inv(A22Inv);
			#ifdef HASCPP11TIMING
			auto t1 = chrono::high_resolution_clock::now();
			#endif

			LU(A22Inv);

			#ifdef HASCPP11TIMING
			auto t2 = chrono::high_resolution_clock::now();
			durationLA_ += chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
			#endif


		#ifdef VERBOSE
			cout << "Finished root level." << endl;
		#endif
			return;
		}

		for (set<int>::iterator it = markedBoxes_.begin(); it != markedBoxes_.end(); it++) {
			int boxIdx = *it;
			Dense<Scalar>& A11     = boxData_[ell][boxIdx].A11post;
			Dense<Scalar>& A22     = boxData_[ell][boxIdx].A22;
			Dense<Scalar>& A22Inv  = boxData_[ell][boxIdx].A22Inv;
			Dense<Scalar>& XL      = boxData_[ell][boxIdx].XL;
			Dense<Scalar>& XR      = boxData_[ell][boxIdx].XR;
			Dense<Scalar>& S       = boxData_[ell][boxIdx].SchurComp;

			// Already pulled up A22 in setup, so just invert it
			A22Inv = A22;
			/* Very possible that XL can't be both LHS and argument */
			Dense<Scalar> A12 = XL, A21 = XR;

			// inv
			// Inv(A22Inv);
			// XL  = A12 * A22Inv;
			// XR  = A22Inv * A21;


			#ifdef HASCPP11TIMING
			auto t1 = chrono::high_resolution_clock::now();
			#endif


			// lu
			LU(A22Inv);
			XL = A12;
			LUSolveT(A22Inv, XL);
			XR = A21;
			LUSolve(A22Inv, XR);

			S    = A12 * XR;
			S    *= -1.0;

			// Update the cliqueified interaction between skeletons
			A11 += S;

			#ifdef HASCPP11TIMING
			auto t2 = chrono::high_resolution_clock::now();
			durationLA_ += chrono::duration_cast<chrono::milliseconds>(t2-t1).count();
			#endif

			IdxVec nborFaces = getNborFacesOfBox_(boxIdx, ell);
			markedFaces_.insert(nborFaces.begin(), nborFaces.end());

		}
		// Finished the box level, push it to the face level
		markedBoxes_.clear();
		set<int> newMarkedFaces;
		//Loop over all faces at this level and populate with info
		for (set<int>::iterator it = markedFaces_.begin(); it != markedFaces_.end(); it++) {
			int faceIdx = *it;
			if (faceIdx == -1) {
				continue;
			}

			Dense<Scalar>& A11  = faceData_[ell-1][faceIdx].A11;
			IdxVec&        self = faceData_[ell-1][faceIdx].DOFs;

			// // In upper levels, replace this with grabbing from children entirely
			// A11 = A_(self,self);

			IdxVec nborBoxes = getNborBoxesOfFace_(faceIdx, ell);
			int boxesPerSide = 1 << ell;
			for (int boxCtr = 0; boxCtr < 2; boxCtr++){
				// grab my diagonal update
				int boxIdx    = nborBoxes[boxCtr];
				IdxVec& bdDOFs   = boxData_[ell][boxIdx].bdDOFs;
				Dense<Scalar>& S = boxData_[ell][boxIdx].SchurComp;

				ScatterAdd_(A11, self, S, bdDOFs);

				// Mark parent boxes of nbor
				TensorIdx3D boxCoord(boxIdx, boxesPerSide, boxesPerSide, boxesPerSide);
				TensorIdx3D parentCoord(boxCoord.x() / 2, boxCoord.y() / 2, boxCoord.z() / 2, boxesPerSide/2, boxesPerSide / 2, boxesPerSide / 2);
				markedBoxes_.insert(parentCoord.LinearIndex());
				// TODO: verify Marking parents faces here
				if (ell > 1) {
					IdxVec faceNbrs = getNborFacesOfBox_(parentCoord.LinearIndex(), ell - 1);
					newMarkedFaces.insert(faceNbrs.begin(), faceNbrs.end());
				}
			}
		}
			//A11.print();

		markedFaces_ = newMarkedFaces;
		// TODO: mark parent faces?

		#ifdef VERBOSE
			cout << "Finished level " << ell <<"." << endl;
		#endif
	}

	template <class Scalar>
	void MF3D<Scalar>::ScatterAdd_(Dense<Scalar>& A, const IdxVec& bigDOFs, const Dense<Scalar>& B, const IdxVec& smallDOFs) {
		// TODO: this is very inefficient (complexity O(c^2))
		IdxVec table1;
		IdxVec table2;

		for (int i = 0; i < smallDOFs.size(); i++){
			int idx = smallDOFs[i];
			 vector<int>::const_iterator it = find(bigDOFs.begin(), bigDOFs.end(), idx);
			 if (it != bigDOFs.end()){
				int newIdx = distance(bigDOFs.begin(), it);
				table1.push_back(newIdx);
				table2.push_back(i);
			}
		}

		for (int i = 0; i < table1.size(); i++){
			for (int j = 0; j < table1.size(); j++) {
				A(table1[i], table1[j]) += B(table2[i],table2[j]);
			}
		}
	}

	template <class Scalar>
	void MF3D<Scalar>::ScatterReplace_(Dense<Scalar>& A, const IdxVec& bigDOFs, const Dense<Scalar>& B, const IdxVec& smallDOFs) {
		// TODO: this is very inefficient (complexity O(c^2))
		IdxVec table1;
		IdxVec table2;

		for (int i = 0; i < smallDOFs.size(); i++){
			int idx = smallDOFs[i];
			 vector<int>::const_iterator it = find(bigDOFs.begin(), bigDOFs.end(), idx);
			 if (it != bigDOFs.end()){
				int newIdx = distance(bigDOFs.begin(), it);
				table1.push_back(newIdx);
				table2.push_back(i);
			}
		}

		for (int i = 0; i < table1.size(); i++){
			for (int j = 0; j < table1.size(); j++) {
				A(table1[i], table1[j]) = B(table2[i],table2[j]);
			}
		}
	}



	template <class Scalar>
	void MF3D<Scalar>::ScatterReplace_(Dense<Scalar>& A, const IdxVec& bigDOFsRow, const IdxVec& bigDOFsCol, const Dense<Scalar>& B, const IdxVec& smallDOFs) {
		// TODO: this is very inefficient (complexity O(c^2))
		IdxVec tableRow1;
		IdxVec tableRow2;
		IdxVec tableCol1;
		IdxVec tableCol2;


		for (int i = 0; i < smallDOFs.size(); i++){
			int idx = smallDOFs[i];
			 vector<int>::const_iterator it = find(bigDOFsRow.begin(), bigDOFsRow.end(), idx);
			 if (it != bigDOFsRow.end()){
				int newIdx = distance(bigDOFsRow.begin(), it);
				tableRow1.push_back(newIdx);
				tableRow2.push_back(i);
			}
			it = find(bigDOFsCol.begin(), bigDOFsCol.end(), idx);
			 if (it != bigDOFsCol.end()){
				int newIdx = distance(bigDOFsCol.begin(), it);
				tableCol1.push_back(newIdx);
				tableCol2.push_back(i);
			}
		}

		for (int i = 0; i < tableRow1.size(); i++){
			for (int j = 0; j < tableCol1.size(); j++) {
				A(tableRow1[i], tableCol1[j]) = B(tableRow2[i],tableCol2[j]);
			}
		}
	}

	template <class Scalar>
	void MF3D<Scalar>::AddNewInteractions_(Dense<Scalar>& A, const IdxVec& totalDOFs, const IdxVec& newDOFs){

		// TODO: this is very inefficient
		IdxVec table;

		vector<bool> flags(totalDOFs.size());

		for (int i = 0; i < newDOFs.size(); i++){
			int idx =  newDOFs[i];
			vector<int>::const_iterator it = find(totalDOFs.begin(), totalDOFs.end(), idx);
			int newIdx = distance(totalDOFs.begin(), it);
			table.push_back(newIdx);
			flags[newIdx] = true;
		}

		for (int i = 0; i < totalDOFs.size(); i++){
			for (int j = 0; j < newDOFs.size(); j++){
				A(i, table[j]) += A_(totalDOFs[i], newDOFs[j]);
				if (table[j] != i && !flags[i]){
					A(table[j], i) += A_(newDOFs[j],totalDOFs[i]);
				}
			}
		}

	}

	template <class Scalar>
	void MF3D<Scalar>::AddNewRowInteractions_(Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colDOFs, const IdxVec& newDOFs){
		// TODO: this is very inefficient
		IdxVec table;

		for (int i = 0; i < newDOFs.size(); i++){
			int idx =  newDOFs[i];
			vector<int>::const_iterator it = find(rowDOFs.begin(), rowDOFs.end(), idx);
			int newIdx = distance(rowDOFs.begin(), it);
			table.push_back(newIdx);
		}

		for (int i = 0; i < newDOFs.size(); i++){
			for (int j = 0; j < colDOFs.size(); j++){
				A(table[i], j) += A_(newDOFs[i], colDOFs[j]);
			}
		}
	}

	template <class Scalar>
	void MF3D<Scalar>::AddNewColInteractions_(Dense<Scalar>& A, const IdxVec& rowDOFs, const IdxVec& colDOFs, const IdxVec& newDOFs){
		// TODO: this is very inefficient
		IdxVec table;

		for (int i = 0; i < newDOFs.size(); i++){
			int idx =  newDOFs[i];
			vector<int>::const_iterator it = find(colDOFs.begin(), colDOFs.end(), idx);
			int newIdx = distance(colDOFs.begin(), it);
			table.push_back(newIdx);
		}

		for (int i = 0; i < rowDOFs.size(); i++){
			for (int j = 0; j < newDOFs.size(); j++){
				A(i, table[j]) += A_(rowDOFs[i], newDOFs[j]);
			}
		}
	}


template <typename Scalar>
Vec<Scalar> MF3D<Scalar>::Apply(const Vec<Scalar>& x) {
	#ifdef VERBOSE
		cout << LINE << endl;
		cout << "Applying MF3D..." << endl;
	#endif


	// Make a copy of x to play with
	Vec<Scalar> b(x);

	// When we apply, we need to hit with the POSITIVE sign on XL, XR
	// Loop over each level from bottom to top
	for (int ell = levels_-1; ell > 0; ell--) {
	#ifdef VERBOSE
		cout << "Upward sweep, level " << ell << endl;
	#endif
		int boxesPerSide = 1 << ell;
		for (int boxIdx = 0; boxIdx < boxesPerSide * boxesPerSide * boxesPerSide; boxIdx++){
			IdxVec &sk         = boxData_[ell][boxIdx].bdDOFs;
			IdxVec &rd         = boxData_[ell][boxIdx].intDOFs;
			Dense<Scalar> &XR  = boxData_[ell][boxIdx].XR;
			Dense<Scalar> &Ard = boxData_[ell][boxIdx].A22;

			Vec<Scalar> b_rd = b(rd);
			Vec<Scalar> b_sk = b(sk);

			b_rd += XR * b_sk;

			b.set(rd, b_rd);

		// Apply redundant diagonal block too
			b_rd = Ard * b_rd;
			b.set(rd, b_rd);
		}
	}

	// Do middle
	#ifdef VERBOSE
	cout << "Middle, level 0" << endl;
	#endif
	IdxVec &rd         = boxData_[0][0].intDOFs;

	Dense<Scalar> &Ard = boxData_[0][0].A22;
	Vec<Scalar> b_rd = b(rd);
	b_rd = Ard * b_rd;
	b.set(rd, b_rd);

	// Loop over each level from top to bottom
	for (int ell = 1; ell < levels_; ell++) {
	#ifdef VERBOSE
		cout << "Downward sweep, level " << ell << endl;
	#endif
		int boxesPerSide = 1 << ell;
		for (int boxIdx = 0; boxIdx < boxesPerSide * boxesPerSide * boxesPerSide; boxIdx++){
			IdxVec &sk         = boxData_[ell][boxIdx].bdDOFs;
			IdxVec &rd         = boxData_[ell][boxIdx].intDOFs;
			Dense<Scalar> &XL  = boxData_[ell][boxIdx].XL;

			Vec<Scalar> b_rd = b(rd);
			Vec<Scalar> b_sk = b(sk);
			b_sk += XL * b_rd;

			b.set(sk, b_sk);
		}
	}

	#ifdef VERBOSE
		cout << "Finished applying MF3D." << endl;
	#endif
		return b;
}

template <typename Scalar>
Vec<Scalar> MF3D<Scalar>::Solve(const Vec<Scalar>& x) {
	#ifdef VERBOSE
		cout << LINE << endl;
		cout << "Solving with MF3D..." << endl;
	#endif
	// Make a copy of x to play with
	Vec<Scalar> b(x);

	// When we solve, we need to hit with the NEGATIVE sign on XL, XR
	// Loop over each level from bottom to top
	for (int ell = levels_-1; ell > 0; ell--) {
	#ifdef VERBOSE
		cout << "Upward sweep, level " << ell << endl;
	#endif
		int boxesPerSide = 1 << ell;
		for (int boxIdx = 0; boxIdx < boxesPerSide * boxesPerSide * boxesPerSide; boxIdx++){
			IdxVec &sk         = boxData_[ell][boxIdx].bdDOFs;
			IdxVec &rd         = boxData_[ell][boxIdx].intDOFs;
			Dense<Scalar> &XL  = boxData_[ell][boxIdx].XL;
			Dense<Scalar> &ArdInv = boxData_[ell][boxIdx].A22Inv;

			Vec<Scalar> b_rd = b(rd);
			Vec<Scalar> b_sk = b(sk);
			b_sk -= XL * b_rd;

			b.set(sk, b_sk);

			// Apply redundant diagonal block too
			// with inv in factor
			//b_rd = ArdInv * b_rd;
			// with lu in factor
			LUSolveVec(ArdInv, b_rd);


			b.set(rd, b_rd);
		}
	}
	// Do middle
	#ifdef VERBOSE
	cout << "Middle, level 0" << endl;
	#endif
	IdxVec &rd         = boxData_[0][0].intDOFs;

	Dense<Scalar> &ArdInv = boxData_[0][0].A22Inv;
	Vec<Scalar> b_rd = b(rd);
	// with inv in factor
	//b_rd = ArdInv * b_rd;
	// with lu in factor
	LUSolveVec(ArdInv, b_rd);
	b.set(rd, b_rd);

	// Loop over each level from top to bottom
	for (int ell = 1; ell < levels_; ell++) {
	#ifdef VERBOSE
		cout << "Downward sweep, level " << ell << endl;
	#endif
		int boxesPerSide = 1 << ell;
		for (int boxIdx = 0; boxIdx < boxesPerSide * boxesPerSide * boxesPerSide; boxIdx++){
			IdxVec &sk         = boxData_[ell][boxIdx].bdDOFs;
			IdxVec &rd         = boxData_[ell][boxIdx].intDOFs;
			Dense<Scalar> &XR  = boxData_[ell][boxIdx].XR;

			Vec<Scalar> b_rd = b(rd);
			Vec<Scalar> b_sk = b(sk);

			b_rd -= XR * b_sk;

			b.set(rd, b_rd);
		}
	}

	#ifdef VERBOSE
		cout << "Finished solving with MF3D." << endl;
	#endif
		return b;
}




}; // namespace TreeFactor3D