#include "vec.hpp"
namespace LinAlg {
	/* Function implementations start here */
	template <class Scalar>
	void Vec<Scalar>::set(const IdxVec& idxs, const Vec& vals) {
		for (int i = 0; i < idxs.size(); i++) {
			#ifdef DEBUG
  				assert(idxs[i] >= 0 && idxs[i] < n_);
  			#endif
  			operator()(idxs[i]) = vals(i);
		}
	}

	template <>
	void Vec<double>::constants_() {
		one_ = 1.0;
		m_one_ = -1.0;
		zero_ = 0.0;

	}
	template <>
	void Vec<dcomplex>::constants_() {
		one_ = dcomplex(1.0);
		m_one_ = dcomplex(-1.0);
		zero_ = dcomplex(0.0);
	}


	template<class Scalar>
	Vec<Scalar>::Vec(int n): n_(n) {
		constants_();
		resize(n);
	}

	template<class Scalar>
	Vec<Scalar>::Vec(const Vec<Scalar>& v): n_(v.n_), data_(v.data_) {
		constants_();
	}


	template <class Scalar>
	Vec<Scalar>::Vec(const Vec<Scalar>& v, const IdxVec& idxs) {
		constants_();
	  	n_ = idxs.size();
	  	resize(n_);
  		for (int i = 0; i < n_; i++) {
  			#ifdef DEBUG
  				assert(idxs[i] >= 0 && idxs[i] < v.n_);
  			#endif
  			operator()(i) = v(idxs[i]);
  		}
  	}


	/* operator overloading */
	template <class Scalar>
	Vec<Scalar>& Vec<Scalar>::operator=(Vec<Scalar> v) {
		if (this != &v) {
			v.swap(*this);
		}
	    return *this;
	}

	template <>
	Vec<double>& Vec<double>::operator+=(const Vec<double>& v) {
		#ifdef DEBUG
			assert(n_ == v.n_);
		#endif
		int ione = 1;
		BLAS(daxpy)(&v.n_, &one_, &v.data_[0], &ione, &data_[0], &ione);
		return *this;
	}

	template <>
	Vec<double>& Vec<double>::operator-=(const Vec<double>& v) {
		#ifdef DEBUG
			assert(n_ == v.n_);
		#endif
		int ione = 1;
		BLAS(daxpy)(&v.n_, &m_one_, &v.data_[0], &ione, &data_[0], &ione);
		return *this;
	}
	template <>
	Vec<dcomplex>& Vec<dcomplex>::operator+=(const Vec<dcomplex>& v) {
		#ifdef DEBUG
			assert(n_ == v.n_);
		#endif
		int ione = 1;
		BLAS(zaxpy)(&v.n_, &one_, &v.data_[0], &ione, &data_[0], &ione);
		return *this;
	}
	template <>
	Vec<dcomplex>& Vec<dcomplex>::operator-=(const Vec<dcomplex>& v) {
		#ifdef DEBUG
			assert(n_ == v.n_);
		#endif
		int ione = 1;
		BLAS(zaxpy)(&v.n_, &m_one_, &v.data_[0], &ione, &data_[0], &ione);
		return *this;
	}

	/* scalar multiplication */
		/* scalar multiplication */
	template <>
	Vec<double>& Vec<double>::operator*=(const double& alpha) {
		int ione = 1;
		BLAS(dscal)(&n_, &alpha, &data_[0], &ione);
		return *this;
	}
	template <>
	Vec<dcomplex>& Vec<dcomplex>::operator*=(const dcomplex& alpha) {
		int ione = 1;
		BLAS(zscal)(&n_, &alpha, &data_[0], &ione);
		return *this;
	}
		template <>
	Vec<double>& Vec<double>::operator/=(const double& alpha) {
		double alpha_i = 1.0 / alpha;
		int ione = 1;
		BLAS(dscal)(&n_, &alpha_i, &data_[0], &ione);
		return *this;
	}
	//TODO: Check if you can divide 1.0 by a dcomplex
	template <>
	Vec<dcomplex>& Vec<dcomplex>::operator/=(const dcomplex& alpha) {
		dcomplex alpha_i = 1.0 / alpha;
		int ione = 1;
		BLAS(zscal)(&n_, &alpha_i, &data_[0], &ione);
		return *this;
	}


	/* frobenius norm */
  	template <>
  	double Vec<double>::norm() const {
  		int ione = 1;
  		return BLAS(dnrm2)(&n_, &data_[0], &ione);
	}
	template <>
  	double Vec<dcomplex>::norm() const {
  		int ione = 1;
  		return BLAS(dznrm2)(&n_, &data_[0], &ione);
	}





	template<class Scalar>
	void Vec<Scalar>::resize(int n) {
		if ( n>0 ) {
			data_.resize(n);
			n_ = n;
		} else if (n < 0) {
			throw std::domain_error("Matrix dimensions must be positive!");
		}
	}


	/* Indexing */
	template <class Scalar>
	inline
	const Scalar& Vec<Scalar>::operator()(int i) const  {
		#ifdef DEBUG
    		assert( i >= 0 && i < n_ );
    	#endif
    	return data_[i];
  	}

  	template <class Scalar>
  	inline
  	Scalar& Vec<Scalar>::operator()(int i)  {
  		#ifdef DEBUG
    		assert( i >= 0 &&  i < n_ );
    	#endif
    	return data_[i];
  	}
  	/* Utility indexing */
  	template <class Scalar>
  	inline
  	Vec<Scalar> Vec<Scalar>::operator()(const IdxVec& idxs) {
  		return Vec<Scalar>(*this, idxs);
  	}

	template <class Scalar>
	void Vec<Scalar>::swap(Vec<Scalar>& that)
    {
        // enable ADL (not necessary in our case, but good practice)
        using std::swap;

        // by swapping the members of two classes,
        // the two classes are effectively swapped

        swap(this->n_, that.n_);
        swap(this->data_, that.data_);

        swap(this->one_, that.one_);
        swap(this->m_one_, that.m_one_);
        swap(this->zero_, that.zero_);
    }



    template <class Scalar>
  	void Vec<Scalar>::print() const {
  		std::cout << n_ << std::endl;

		for(int i = 0; i < n_; i++) {
			Scalar v = operator()(i);
  			v = fabs(v) > 1e-8? v : 0;
	  		std::cout << " " << v;
		}
			std::cout << std::endl;
	}


}; // namespace LinAlg