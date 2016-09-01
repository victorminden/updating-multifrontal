#include "sparse.hpp"
namespace LinAlg {

	// initialize with a file of CSR
	template<class Scalar>
	Sparse<Scalar>::Sparse(const std::string& filename) {
		try {
			int nnz, v;
			double a;
			std::ifstream file;
			file.open(filename);

			if (!file.is_open()) {
				throw std::runtime_error("failed to open file");
			}

			file >> m_;
			file >> n_;
			file >> nnz;

			val_.resize(nnz);
			col_ind_.resize(nnz);
			row_ptr_.resize(m_+1);

			for (size_t i = 0; i < nnz; i++) {
				file >> a;
				val_[i] = a;
			}
			for (size_t i = 0; i < nnz; i++) {
				file >> v;
				col_ind_[i] = v;
			}
			for (size_t i = 0; i < m_+1; i++) {
				file >> v;
				row_ptr_[i] = v;
			}
			nnz_ = v;

		} catch (int e) {
			std::cout << "Failed to open file!" << std::endl;
			exit(1);
		}

	}

	template<class Scalar>
	Sparse<Scalar>::Sparse(const std::vector<Scalar>& val, const IdxVec& col_ind,
		const IdxVec& row_ptr, int m, int n) : m_(m), n_(n),
												val_(val), col_ind_(col_ind),
												row_ptr_(row_ptr) {
		#ifdef DEBUG
			assert(m_ + 1 == row_ptr_.size());
		#endif
		nnz_ = row_ptr[m_];
	}

	template <class Scalar>
	Sparse<Scalar>::Sparse(const Sparse& other)
	: m_(other.m_), n_(other.n_), nnz_(other.nnz_), val_(other.val_),col_ind_(other.col_ind_), row_ptr_(other.row_ptr_){}


	/* CSR MatVec */
	template <class Scalar>
	Vec<Scalar> Sparse<Scalar>::operator*(const Vec<Scalar>& v) const {
		Vec<Scalar> y = Vec<Scalar>(m_);
		#ifdef DEBUG
			assert(v.n_ == m_);
		#endif
		for (int i = 0; i < m_; i++) {
				y(i) = 0;
			for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
				int j = col_ind_[idx];
				y(i) += val_[idx] * v(j);
			}
		}
		return y;
	}

	/* Indexing */
	template <class Scalar>
	const Scalar Sparse<Scalar>::operator()(int i, int j) const {
		#ifdef DEBUG
			assert(i >= 0 && i < m_ && j >= 0 && j < n_);
		#endif
		for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
			int curr_j = col_ind_[idx];
			if (curr_j == j) {
				return val_[idx];
			}
		}
		return 0;
	}


	template <class Scalar>
	Dense<Scalar> Sparse<Scalar>::operator()(const IdxVec& rows, const IdxVec& cols) const {
		int m = rows.size();
		int n = cols.size();
		Dense<Scalar> B(m,n);

		for (int ii = 0; ii < rows.size(); ii++) {
			for (int jj = 0; jj < cols.size(); jj++) {
				int i = rows[ii], j = cols[jj];
				#ifdef DEBUG
					assert(i >= 0 && i < m_ && j >= 0 && j < n_);
				#endif
				for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
					int curr_j = col_ind_[idx];
					if (curr_j == j) {
						B(ii,jj) = val_[idx];
					}
				}
			}
		}
		return B;
	}

	// TODO: CHECK ME
	template <class Scalar>
	Dense<Scalar> Sparse<Scalar>::operator()(const IdxVec& rows, const IdxVec& cols, bool vstack) const {
		int m = rows.size();
		int n = cols.size();
		Dense<Scalar> B(2*m,n);

		for (int ii = 0; ii < rows.size(); ii++) {
			for (int jj = 0; jj < cols.size(); jj++) {
				int i = rows[ii], j = cols[jj];
				#ifdef DEBUG
					if (!(i >= 0 && i < m_ && j >= 0 && j < n_)) {
						for (int i = 0; i < cols.size(); i ++) {
							std::cout << cols[i] << std::endl;
						}
					}
					assert(i >= 0 && i < m_ && j >= 0 && j < n_);
				#endif
				for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
					int curr_j = col_ind_[idx];
					if (curr_j == j) {
						B(ii,jj) = val_[idx];
					}
				}
			}
		}

		// Now grab the transpose by switching i and j
		for (int ii = 0; ii < rows.size(); ii++) {
			for (int jj = 0; jj < cols.size(); jj++) {
				int j = rows[ii], i = cols[jj];
				#ifdef DEBUG
					if (!(i >= 0 && i < m_ && j >= 0 && j < n_)) {
						std::cout << i << ' ' << j << ' ' << m_ << ' ' << n_ << std::endl;
					}
					assert(i >= 0 && i < m_ && j >= 0 && j < n_);
				#endif
				for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
					int curr_j = col_ind_[idx];
					if (curr_j == j) {
						B(m+ii,jj) = val_[idx];
					}
				}
			}
		}

		return B;
	}

	/* Frobenius norm */
  	template <>
  	double Sparse<double>::norm() const {
  		int ione = 1;
  		return BLAS(dnrm2)(&row_ptr_[m_], &val_[0], &ione);
	}
	template <>
  	double Sparse<dcomplex>::norm() const {
  		int ione = 1;
  		return BLAS(dznrm2)(&row_ptr_[m_], &val_[0], &ione);
	}

	/* Printing */
	template <class Scalar>
	void Sparse<Scalar>::print() const {
		std::cout << m_ <<" " << n_ << std::endl;
		for (int i = 0; i < m_; i++) {
			for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
				int j = col_ind_[idx];
				std::cout << "(i,j,a) = (" << i <<"," << j << "," << val_[idx] << ")" << std::endl;
			}
		}
		std::cout << std::flush;
	}

	template <class Scalar>
	void Sparse<Scalar>::printDense() const {
		std::cout << m_ <<" " << n_ << std::endl;
		for (int i = 0 ; i < m_; i++){
			for (int j = 0; j < n_; j++) {
				std::cout << operator()(i,j) << ' ';
			}
			std::cout << std::endl;
		}
	}

	template <class Scalar>
	void Sparse<Scalar>::printMatlab() const {
		std::ofstream myfile;
  		myfile.open ("data.csv");
  		//myfile << m_ << ',' << n_ << ',';
		for (int i = 0; i < m_; i++) {
			for (int idx = row_ptr_[i]; idx < row_ptr_[i+1]; idx++) {
				int j = col_ind_[idx];
				myfile << i +1 <<"," << j + 1 << "," << val_[idx] << ",";
			}
		}
		myfile.close();
		//myfile << "]; i =data(1:3:end); j=data(2:3:end); a = data(3:3:end); A = sparse(i,j,a,m,n);" << std::flush;
	}


}; // namespace LinAlg