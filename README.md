# README #
Joint work with Anil Damle, Ken Ho, and Lexing Ying.

Based on the updating technique described by [Minden et al.] (http://dx.doi.org/10.1137/15M1024500) for the hierarchical interpolative factorization, we implement a method for updating factorizations from the  multifrontal method with nested dissection ordering for discretizations of elliptic PDEs in 2D or 3D.  

On its own, this updating scheme applied to the multifrontal method should not attain greater than a constant factor speed-up compared to updating anew, but I have received requests that this implementation be made available.  A corresponding description of the initial implementation is available in my class project write-up [here](https://victorminden.github.io/docs/project_cme335.pdf).

This is primarily personal research code and therefore is provided as-is with minimal documentation.  Further, as it is stepping-stone code on the way to more complicated methods, it is not necessarily the most efficient when applied specifically to the multifrontal case.


### How do I get set up? ###

mkdir build

cd build

cmake ..

make

### What are the tests? ###

Given a discretization of the Laplacian on a unit box (square/cube), the tests in updating-multifrontal/test_src/DE perform a multifrontal factorization first for the constant case and then update the resulting factorization to reflect a localized change in the coefficients.
