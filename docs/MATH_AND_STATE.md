# What the code's state and matrices mean

For a history `h`, the teacher's relevant inference state contains its per-layer K/V tensors and position/mask bookkeeping. Current residual vectors can be measured too, but they are not independent variables passed into the next cached step. The proposal tests three **selected sources** of information, not an assertion that any source equals the complete teacher state.

The fitted pipeline is

`selected teacher numbers → frozen standardization/PCA → learned encoder → compact state`.

For the density model the encoder emits three numbers `u`. It bounds them by

\[
r=\frac{u}{\sqrt{1+\|u\|^2}},\qquad
\rho=\frac12\begin{pmatrix}1+r_z&r_x-i r_y\\r_x+i r_y&1-r_z\end{pmatrix}.
\]

This construction guarantees a Hermitian, positive-semidefinite, trace-one matrix in exact arithmetic. The two eigenvalues are `(1±||r||)/2`. It does **not** guarantee that the matrix predicts anything correctly. Its three real coordinates, observable stage and position form the tested compact state.

For each token, the learned operators give an unnormalized child matrix

\[
A_x(\rho)=\sum_jK_{xj}\rho K_{xj}^\dagger.
\]

The probability is its trace; the updated matrix is `A_x(rho)/trace(A_x(rho))`. Normalizing the raw operators so that `sum K†K=I` makes all eight probabilities sum to one. The equivalent readout matrix is `E_x=sum_j K†_xj K_xj`, so `q(x)=trace(E_x rho)`. This is why the density model needs no extra probability network.

The entries are not automatically token probabilities. The diagonal entries would be probabilities of a particular basis measurement. The **learned token measurement** here uses `E_x`; token probabilities are its trace products. Off-diagonal entries encode other fitted coordinates and are not independently identified causes. Different simultaneous changes of matrix/operator coordinates can represent the same behavior.

The known generator has a simpler exact reference: store `b=P(Z=1|history)`. Waiting fillers keep b, reset sets b=1/2, and a query exposes b as the next answer-1 probability. The diagonal matrix `diag(1-b,b)` is one valid realization. `reference.py` implements it for correctness checks; the fitted encoder never receives b or these known operators as its solution.

Trace distance between two 2×2 states is `||r-r'||/2`. For common stage and position, their **complete joint** surrogate future distributions have TV at most that distance. Teacher-future TV can exceed it by the two surrogate approximation errors:

\[
\operatorname{TV}(Q_T^k(h),Q_T^k(h'))\leq
\operatorname{TV}(Q_T^k(h),Q_D^k(h))+
D_{tr}(\rho_h,\rho_{h'})+
\operatorname{TV}(Q_D^k(h'),Q_T^k(h')).
\]

This bound is checked against measured complete short trees. It is not a bound on the predictions after conditioning on an arbitrary rare supplied string. The worked example has two-token TV .06, while conditioning on the common query yields answer TV .6. Conditioning can amplify a difference.

The experiment succeeds only through measured prediction and intervention performance under its predefined gates. A valid matrix, a visually appealing ball, a high probe score, or agreement on the most likely token alone does not establish the proposed causal summary.
