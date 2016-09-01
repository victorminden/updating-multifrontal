#include "dense.hpp"
namespace LinAlg {
	/* Function declarations start here */
	template <class Scalar>
	void Dense<Scalar>::swap(Dense<Scalar>& that)
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two classes,
        // the two classes are effectively swapped
        swap(this->m_, that.m_);
        swap(this->n_, that.n_);
        swap(this->data_, that.data_);
        swap(this->size_, that.size_);
        swap(this->ipiv_, that.ipiv_);
    }

	template <class Scalar>
	void Dense<Scalar>::resize(int m, int n) {
		size_ = m * n;
		m_ = m;
		n_ = n;
		if (size_ > 0 && m > 0 && n > 0) {
			data_.resize(m * n);
		} else if (m < 0 || n < 0) {
			throw std::domain_error("Matrix dimensions must be positive!");
		}
	}

	template <>
	void Dense<double>::constants_() {
		one_   = 1.0;
		m_one_ = -1.0;
		zero_  = 0.0;

	}
	template <>
	void Dense<dcomplex>::constants_() {
		one_   = dcomplex(1.0);
		m_one_ = dcomplex(-1.0);
		zero_  = dcomplex(0.0);
	}

	template <class Scalar>
	Dense<Scalar>::Dense(int m, int n): m_(m), n_(n) {
		// #ifdef VERBOSE
		// 	std::cout << "Called two-argument constructor" << std::endl;
		// #endif
		size_  = m * n;
		constants_();
		if (size_ > 0 && m > 0 && n > 0) {
			data_.resize(m * n);
		} else if (m < 0 || n < 0) {
			throw std::domain_error("Matrix dimensions must be positive!");
		}
	}

	template <class Scalar>
	Dense<Scalar>::Dense(const Dense<Scalar>& A): m_(A.m_), n_(A.n_), size_(A.size_), data_(A.data_), ipiv_(A.ipiv_){
		constants_();
		// #ifdef VERBOSE
		// 	std::cout << "Called deep copy constructor!" << std::endl;
		// #endif
	}

	template <class Scalar>
  	Dense<Scalar>::Dense(const Dense<Scalar>& A, const IdxVec& rows, const IdxVec& cols) {
  		constants_();
  // 		#ifdef VERBOSE
		// 	std::cout << "Called submat constructor!" << std::endl;
		// #endif
  		m_ = rows.size();
  		n_ = cols.size();
  		size_ = m_ * n_;
  		resize(m_, n_);
  		for (int j = 0; j < n_; j++) {
  			for (int i = 0; i < m_; i++) {
  				int ii = rows[i], jj = cols[j];
  				#ifdef DEBUG
  					assert(ii >= 0 && ii < A.m_ && jj >= 0 && jj < A.n_);
  				#endif
  				operator()(i,j) = A(ii,jj);
  			}
  		}
  	}

	/* Overloaded operators */
	template <class Scalar>
	Dense<Scalar>& Dense<Scalar>::operator=(Dense<Scalar> A) {
		// #ifdef VERBOSE
		// 	std::cout << "Called assignment operator" << std::endl;
		// #endif
		if (this != &A) {
			A.swap(*this);
		}
	    return *this;
	}


	/* addition / subtraction */
	template <>
	Dense<double>& Dense<double>::operator+=(const Dense<double>& A) {
		#ifdef DEBUG
			assert(n_ == A.n_);
    		assert(m_ == A.m_);
    	#endif
		int ione = 1;
		BLAS(daxpy)(&A.size_ , &one_, &A.data_[0], &ione, &data_[0], &ione);
		return *this;
	}

	template <>
	Dense<double>& Dense<double>::operator-=(const Dense<double>& A) {
		#ifdef DEBUG
			assert(n_ == A.n_);
    		assert(m_ == A.m_);
    	#endif

		int ione = 1;
		BLAS(daxpy)(&A.size_, &m_one_, &A.data_[0], &ione, &data_[0], &ione);
		return *this;
	}
	template <>
	Dense<dcomplex>& Dense<dcomplex>::operator+=(const Dense<dcomplex>& A) {
		#ifdef DEBUG
			assert(n_ == A.n_);
    		assert(m_ == A.m_);
    	#endif

		int ione = 1;
		BLAS(zaxpy)(&A.size_ , &one_, &A.data_[0], &ione, &data_[0], &ione);
		return *this;
	}
	template <>
	Dense<dcomplex>& Dense<dcomplex>::operator-=(const Dense<dcomplex>& A) {
		#ifdef DEBUG
			assert(n_ == A.n_);
    		assert(m_ == A.m_);
    	#endif

		int ione = 1;
		BLAS(zaxpy)(&A.size_, &m_one_, &A.data_[0], &ione, &data_[0], &ione);
		return *this;
	}


	/* Workhorse routines */
	// This might be a slow dispatch, should time it to see if it's worth it at all

	template <>
	Dense<double> Dense<double>::operator*(const Dense<double>& B) const {
		// C = A * B
		#ifdef DEBUG
			assert(n_ == B.m_);
		#endif
		char Aflag = 'N', Bflag = 'N';
		Dense<double> C(m_, B.n_);
		if (m_ != 0 && n_ != 0  && B.n_ != 0) {
			BLAS(dgemm)(&Aflag, &Bflag, &m_, &B.n_, &n_, &one_, &data_[0], &m_, &B.data_[0], &B.m_, &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}

	template <>
	Dense<double> Dense<double>::operator*(const TransposeView_<double>& B) const {
		// C = A * B^(T,C)
		#ifdef DEBUG
			assert(n_ == B.mat_->n_);
		#endif
		char Aflag = 'N', Bflag = B.transA_;
		Dense<double> C(m_, B.mat_->m_);
		if (m_ != 0 && n_ != 0 && B.mat_->m_ != 0) {
			BLAS(dgemm)(&Aflag, &Bflag, &m_, &(B.mat_->m_), &n_, &one_, &data_[0], &m_, &(B.mat_->data_[0]), &(B.mat_->m_), &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}

	template <>
	Dense<dcomplex> Dense<dcomplex>::operator*(const Dense<dcomplex>& B) const {
		// C = A * B
		#ifdef DEBUG
			assert(n_ == B.m_);
		#endif
		char Aflag = 'N', Bflag = 'N';
		Dense<dcomplex> C(m_, B.n_);
		if (m_ != 0 && n_ != 0  && B.n_ != 0) {
			BLAS(zgemm)(&Aflag, &Bflag, &m_, &B.n_, &n_, &one_, &data_[0], &m_, &B.data_[0], &B.m_, &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}

	template <>
	Dense<dcomplex> Dense<dcomplex>::operator*(const TransposeView_<dcomplex>& B) const {
		// C = A * B^(T,C)
		#ifdef DEBUG
			assert(n_ == B.mat_->n_);
		#endif
		char Aflag = 'N', Bflag = B.transA_;
		Dense<dcomplex> C(m_, B.mat_->m_);
		if (m_ != 0 && n_ != 0 && B.mat_->m_ != 0) {
			BLAS(zgemm)(&Aflag, &Bflag, &m_, &(B.mat_->m_), &n_, &one_, &data_[0], &m_, &(B.mat_->data_[0]), &(B.mat_->m_), &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}


	template <>
	Vec<double> Dense<double>::operator*(const Vec<double>& v) const {
		char Aflag = 'N';
		#ifdef DEBUG
			assert(v.n_ == n_);
		#endif
		Vec<double> y(m_);
		int onei = 1;
		BLAS(dgemv)(&Aflag, &m_, &n_, &one_, &data_[0], &m_, &v.data_[0], &onei, &zero_, &y.data_[0], &onei);
		return y;
	}


	template <>
	Vec<dcomplex> Dense<dcomplex>::operator*(const Vec<dcomplex>& v) const {
		char Aflag = 'N';
		#ifdef DEBUG
			assert(v.n_ == n_);
		#endif
		Vec<dcomplex> y(m_);
		int onei = 1;
		BLAS(zgemv)(&Aflag, &m_, &n_, &one_, &data_[0], &m_, &v.data_[0], &onei, &zero_, &y.data_[0], &onei);
		return y;
	}




	template <class Scalar>
		TransposeView_<Scalar> Dense<Scalar>::T() {
		return TransposeView_<Scalar>(*this, 'T');
	}

	template <class Scalar>
		TransposeView_<Scalar> Dense<Scalar>::H() {
		return TransposeView_<Scalar>(*this, 'C');
	}


	/* scalar multiplication */
	template <>
	Dense<double>& Dense<double>::operator*=(const double& alpha) {
		int ione = 1;
		BLAS(dscal)(&size_, &alpha, &data_[0], &ione);
		return *this;
	}
	template <>
	Dense<dcomplex>& Dense<dcomplex>::operator*=(const dcomplex& alpha) {
		int ione = 1;
		BLAS(zscal)(&size_, &alpha, &data_[0], &ione);
		return *this;
	}
		template <>
	Dense<double>& Dense<double>::operator/=(const double& alpha) {
		double alpha_i = 1.0 / alpha;
		int ione = 1;
		BLAS(dscal)(&size_, &alpha_i, &data_[0], &ione);
		return *this;
	}
	//TODO: Check if you can divide 1.0 by a dcomplex
	template <>
	Dense<dcomplex>& Dense<dcomplex>::operator/=(const dcomplex& alpha) {
		dcomplex alpha_i = 1.0 / alpha;
		int ione = 1;
		BLAS(zscal)(&size_, &alpha_i, &data_[0], &ione);
		return *this;
	}


	/* frobenius norm */
  	template <>
  	double Dense<double>::norm() const {
  		int ione = 1;
  		return BLAS(dnrm2)(&size_, &data_[0], &ione);
	}
	template <>
  	double Dense<dcomplex>::norm() const {
  		int ione = 1;
  		return BLAS(dznrm2)(&size_, &data_[0], &ione);
	}



	/* Indexing */
	template <class Scalar>
	inline
	const Scalar& Dense<Scalar>::operator()(int i, int j) const  {
		#ifdef DEBUG
    		assert( i >= 0 && i < m_ && j >= 0 && j < n_ );
    	#endif
    	return data_[i + j * m_];
  	}

  	template <class Scalar>
  	inline
  	Scalar& Dense<Scalar>::operator()(int i, int j)  {
  		#ifdef DEBUG
    		assert( i >= 0 && i < m_ && j >= 0 && j < n_ );
    	#endif
    	return data_[i + j * m_];
  	}

  	/* Utility indexing */

  	// TODO: This could maybe be faster (not for sure, but maybe)
  	template <class Scalar>
  	inline
  	Dense<Scalar> Dense<Scalar>::operator()(const IdxVec& rows, const IdxVec& cols) {
  		return Dense<Scalar>(*this, rows, cols);
  	}

  	/* Printing */
  	template <class Scalar>
  	void Dense<Scalar>::print() const {
		std::cout << m_ << " " << n_ << std::endl;
		for(int i = 0; i < m_; i++) {
			for(int j = 0; j < n_; j++) {
	  			std::cout << " " <<  std::fixed << std::setprecision(4) << std::setw(8) << operator()(i,j);
			}
			std::cout << std::endl;
		}
  	}


/* Interpolative decomposition */
/* -------------------------------------------------------------- */

	void InterpDecomp(Dense<double> A, Dense<double>& T, IdxVec& sk, IdxVec& rd, double tol)
	{
	  /*
	   computes an interpolative factorization such that A[:,rd] \approx A[:,sk]*T
	   this will break if the matrix A is all 0 or tol > 1
	  */
	  int m = A.m_;
	  int n = A.n_;
	  int nsk = 0;
	  int nrd = 0;
	  assert( m>0 );
	  assert( n>0 );
	  int k = std::min(m,n);
	  nsk = k;
	  double wkopt;
	  int info;
	  char UT = 'U';
	  char NO = 'N';
	  std::vector<double> tau(k);
	  IdxVec pvt(n);
	  double normtol;

	  /* Set up pvt so that all of the columns are free to be pivoted */
	  // for(int i=0; i<n; i++)
	  // {
	  //   pvt[i] = 0;
	  // }
	  /* First half of QR */
	  /* Query workspace */
	  int lwork = -1;
	  LAPACK(dgeqp3)( &m, &n, &A.data_[0], &m, &pvt[0], &tau[0], &wkopt, &lwork, &info);
	  lwork = (int)(wkopt+0.5);
	  assert(lwork>0);
	  std::vector<double>  work1(lwork);
	  /* compute */
	  LAPACK(dgeqp3)( &m, &n, &A.data_[0], &m, &pvt[0], &tau[0], &work1[0], &lwork, &info);
	  assert(info==0);

	  /* Read out R and stop once the error critera are met */
	  /* Also, this will break if tol > 1 because R will not have to exist by this error metric */
	  assert( tol <= 1);
	  normtol = std::abs(A(0,0)*tol);

	  for(int i=1;i<k; i++){
	    if (std::abs(A(i,i)) < normtol)
	    {
	      nsk = i;
	      break;

	    }
	  }


	  /*
	    Maybe there is some way to get around this,
	    but the sizes are not known ahead of time.
	    Could ask for slightly larger sk and rd and also return nsk and nrd.
	  */
	  nrd = n-nsk;
	  sk.resize(nsk);
	  rd.resize(nrd);
	  T.resize(nsk,nrd);


	  /* Extract sk and rd comlumns */
	  /* The -1 makes the indices start at 0 */
	  for(int i=0; i<nsk; i++)
	  {
	    sk[i] = pvt[i]-1;
	  }
	  for(int i=nsk; i<n; i++)
	  {
	    rd[i-nsk] = pvt[i]-1;
	  }

	  if (nrd > 0)
	  {
	    LAPACK(dtrtrs)( &UT, &NO, &NO, &nsk, &nrd, &A.data_[0], &m, &A.data_[nsk*m], &m, &info);
	    assert(info == 0);
	    for (int i=0; i<nrd; i++)
	    {
	      memcpy(&T.data_[i*nsk], &A.data_[(nsk+i)*m],nsk*sizeof(double));
	    }
	  }

	}

	void InterpDecomp(Dense<dcomplex> A, Dense<dcomplex>& T, IdxVec& sk, IdxVec& rd, double tol)
	{

	  // computes an interpolative factorization such that A[:,rd] \approx A[:,sk]*T
	  int m = A.m_;
	  int n = A.n_;
	  int nsk = 0;
	  int nrd = 0;
	  assert( m>0 && n>0 );
	  int k = std::min(m,n);
	  nsk = k;
	  dcomplex wkopt;
	  int info;
	  char UT = 'U';
	  char NO = 'N';
	  std::vector<dcomplex> tau(k);
	  IdxVec pvt(n);
	  std::vector<double> rwork(2*n);
	  double normtol = 0.0;
	  /* Set up pvt so that all of the columns are free to be pivoted */
	  // for(int i=0; i<n; i++)
	  // {
	  //   pvt[i] = 0;
	  // }
	  /* First half of QR */
	  /* Query workspace */
	  int lwork = -1;
	  double dblwork;
	  LAPACK(zgeqp3)( &m, &n, &A.data_[0], &m, &pvt[0], &tau[0], &wkopt, &lwork, &rwork[0], &info);

	  // This is a complete workaround at the moment
	  dblwork = std::abs(wkopt)+0.5;
	  lwork = (int)dblwork;

	  assert(lwork>0);
	  std::vector<dcomplex> work1;
	  work1.reserve(lwork);
	  /* compute */
	  LAPACK(zgeqp3)( &m, &n, &A.data_[0], &m, &pvt[0], &tau[0], &work1[0], &lwork, &rwork[0], &info);
	  assert(info==0);

	  /* Read out R and stop once the error critera are met */
	  /* TODO: Change the error criteria to match the python code, this is just for testing...*/
	  /* Also, this will break if tol > 1 because R will not have to exist by this error metric */
	  normtol = std::abs(A(0,0)*tol);
	  assert( tol <= 1);
	  for(int i=1;i<k; i++){
	    if (std::abs(A(i,i)) < normtol)
	    {
	      nsk = i;
	      break;
	    }
	  }

	  /*
	    Maybe there is some way to get around this,
	    but the sizes are not known ahead of time.
	    Could ask for slightly larger sk and rd and also return nsk and nrd.
	  */
	  nrd = n-nsk;
	  sk.resize(nsk);
	  rd.resize(nrd);
	  T.resize(nsk,nrd);

	  /* Extract sk and rd comlumns */
	  /* The -1 makes the indices start at 0 */
	  for(int i=0; i<nsk; i++)
	  {
	    sk[i] = pvt[i]-1;
	  }
	  for(int i=nsk; i<n; i++)
	  {
	    rd[i-nsk] = pvt[i]-1;
	  }

	  if (nrd > 0)
	  {
	    LAPACK(ztrtrs)( &UT, &NO, &NO, &nsk, &nrd, &A.data_[0], &m, &A.data_[nsk*m], &m, &info);

	    for (int i=0; i<nrd; i++)
	    {
	      memcpy(&T.data_[i*nsk], &A.data_[(nsk+i)*m],nsk*sizeof(dcomplex));
	    }
	  }

	}




/* -------------------------------------------------------------- */
	/* TransposeView_ */
  	template <class Scalar>
  	TransposeView_<Scalar>::TransposeView_(const Dense<Scalar>& A, char transA) {
  		mat_ = &A;
  		transA_ = transA;
  		buffer_ = &A.data_[0];
  		one_ = A.one_;
  		m_one_ = A.m_one_;
  		zero_ = A.zero_;
  	}



  	template <>
	Dense<double> TransposeView_<double>::operator*(const Dense<double>& B) const {
		// C = A^(T,C) * B
		#ifdef DEBUG
			assert(mat_->m_ == B.m_);
		#endif
		char Aflag = transA_, Bflag = 'N';
		Dense<double> C(mat_->n_, B.n_);
		if (B.m_ != 0 && mat_->n_ != 0 && B.n_ != 0) {
			BLAS(dgemm)(&Aflag, &Bflag, &(mat_->n_), &B.n_, &(mat_->m_), &one_, &(mat_->data_[0]), &(mat_->m_), &B.data_[0], &B.m_, &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}


	template <>
	Dense<dcomplex> TransposeView_<dcomplex>::operator*(const Dense<dcomplex>& B) const {
		// C = A^(T,C) * B
		#ifdef DEBUG
			assert(mat_->m_ == B.m_);
		#endif
		char Aflag = transA_, Bflag = 'N';
		Dense<dcomplex> C(mat_->n_, B.n_);
		if (B.m_ != 0 && mat_->n_ != 0 && B.n_ != 0) {
			BLAS(zgemm)(&Aflag, &Bflag, &(mat_->n_), &B.n_, &(mat_->m_), &one_, &(mat_->data_[0]), &(mat_->m_), &B.data_[0], &B.m_, &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}



	template <>
	Dense<double> TransposeView_<double>::operator*(const TransposeView_<double>& B) const {
		// C = A^(T,C) * B^(T,C)
		#ifdef DEBUG
			assert(mat_->m_ == B.mat_->n_);
		#endif
		char Aflag = transA_, Bflag = B.transA_;
		Dense<double> C(mat_->n_, B.mat_->m_);
		if ( mat_->m_ != 0 && mat_-> n_ != 0 && B.mat_->m_ != 0) {
			BLAS(dgemm)(&Aflag, &Bflag, &(mat_->n_), &(B.mat_->m_), &(mat_->m_), &one_, &(mat_->data_[0]), &(mat_->m_), &(B.mat_->data_[0]), &(B.mat_->m_), &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}



	template <>
	Dense<dcomplex> TransposeView_<dcomplex>::operator*(const TransposeView_<dcomplex>& B) const {
		// C = A^(T,C) * B^(T,C)
		#ifdef DEBUG
			assert(mat_->m_ == B.mat_->n_);
		#endif
		char Aflag = transA_, Bflag = B.transA_;
		Dense<dcomplex> C(mat_->n_, B.mat_->m_);
		if ( mat_->m_ != 0 && mat_-> n_ != 0 && B.mat_->m_ != 0) {
			BLAS(zgemm)(&Aflag, &Bflag, &(mat_->n_), &(B.mat_->m_), &(mat_->m_), &one_, &(mat_->data_[0]), &(mat_->m_), &(B.mat_->data_[0]), &(B.mat_->m_), &zero_, &C.data_[0], &C.m_);
		}
		return C;
	}


	template <>
	Vec<double> TransposeView_<double>::operator*(const Vec<double>& v) const {
		char Aflag = transA_;
		#ifdef DEBUG
			assert(v.n_ == mat_->m_);
		#endif
		Vec<double> y(mat_->n_);
		int onei = 1;
		BLAS(dgemv)(&Aflag, &(mat_->m_), &(mat_->n_), &one_, &(mat_->data_[0]), &(mat_->m_), &v.data_[0], &onei, &zero_, &y.data_[0], &onei);
		return y;
	}


	template <>
	Vec<dcomplex> TransposeView_<dcomplex>::operator*(const Vec<dcomplex>& v) const {
		char Aflag = transA_;
		#ifdef DEBUG
			assert(v.n_ == mat_->m_);
		#endif
		Vec<dcomplex> y(mat_->n_);
		int onei = 1;
		BLAS(zgemv)(&Aflag, &(mat_->m_), &(mat_->n_), &one_, &(mat_->data_[0]), &(mat_->m_), &v.data_[0], &onei, &zero_, &y.data_[0], &onei);
		return y;
	}




	template <>
  	void TransposeView_<double>::print() const {
  		int n_ = mat_->n_, m_ = mat_->m_;
  		std::cout << n_ << " " << m_ << std::endl;
		for(int j = 0; j < n_; j++) {
			for(int i = 0; i < m_; i++) {
		  		std::cout << " "  << std::fixed << std::setprecision(4) << std::setw(8) << (*mat_)(i,j);
			}
			std::cout << std::endl;
		}
  	}

  	template <>
  	void TransposeView_<dcomplex>::print() const {
		if (transA_ == 'T') {
			int n_ = mat_->n_, m_ = mat_->m_;
			std::cout << n_ << " " << m_ << std::endl;
			for(int j = 0; j < n_; j++) {
				for(int i = 0; i < m_; i++) {
		  			std::cout << " " << std::fixed << std::setprecision(4) << std::setw(8) << (*mat_)(i,j);
				}
				std::cout << std::endl;
			}
		} else {
			int n_ = mat_->n_, m_ = mat_->m_;
			std::cout << n_ << " " << m_ << std::endl;
			for(int j = 0; j < n_; j++) {
				for(int i = 0; i < m_; i++) {
		  			std::cout << " "  << std::fixed << std::setprecision(4) << std::setw(8) << std::conj((*mat_)(i,j));
				}
				std::cout << std::endl;
			}
		}
  	}



  }; // namespace LinAlg
