# The barrier method (for Quadratic optimization problem) in Python
Basic Python implementation of the barrier method to solve a Quadratic optimization Problem.

Implemented for the course "Convex optimization and applications in machine learning" for Master MVA.

We consider the (QP) problem:
minimize <img src="https://render.githubusercontent.com/render/math?math=v^TQv+"> + <img src="https://render.githubusercontent.com/render/math?math=p^Tv">
subject to <img src="https://render.githubusercontent.com/render/math?math=Av \preceq b">
With <img src="https://render.githubusercontent.com/render/math?math=v \in R^n"> and where <img src="https://render.githubusercontent.com/render/math?math=Q \succeq 0">.


The function <code>v_seq = centering_step(Q,p,A,b,t,v0,eps)</code> implements
the Newton method to solve the centering step given the inputs (Q, p, A, b), the
barrier method parameter t, initial variable <img src="https://render.githubusercontent.com/render/math?math=v_0"> and a target precision ![\epsilon](https://render.githubusercontent.com/render/math?math=%5Cepsilon) The function outputs the sequence of variables iterates <img src="https://render.githubusercontent.com/render/math?math=(vi)_{i=1,...,n_\epsilon}">, where ![\epsilon](https://render.githubusercontent.com/render/math?math=n_%5Cepsilon) is the number of iterations to obtain the ![\epsilon](https://render.githubusercontent.com/render/math?math=%5Cepsilon) precision, using a backtracking line search.

The function <code>v_seq = barr_method(Q,p,A,b,v0,eps)</code> which implements the barrier method to solve (QP) using precedent function given the data inputs (Q, p, A, b), a feasible point v0, a precision criterion ![\epsilon](https://render.githubusercontent.com/render/math?math=%5Cepsilon). The function outputs the sequence of variables iterates <img src="https://render.githubusercontent.com/render/math?math=(vi)_{i=1,...,n_\epsilon}">, where ![\epsilon](https://render.githubusercontent.com/render/math?math=n_%5Cepsilon) is the number of iterations to obtain the ![\epsilon](https://render.githubusercontent.com/render/math?math=%5Cepsilon) precision.

We test our function on randomly generated matrices X and observations y with λ = 10.
We plot precision criterion and gap ![*](https://render.githubusercontent.com/render/math?math=f(v_t)-f^*) in semilog scale (using the best value found for ![\epsilon](https://render.githubusercontent.com/render/math?math=f) as a surrogate for ![*](https://render.githubusercontent.com/render/math?math=f^*)). We repeat for different values of the barrier method parameter µ = 2, 15, 50, 100,... and check the impact on w.
