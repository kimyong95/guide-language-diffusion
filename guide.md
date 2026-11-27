## M (Distribution Selection)

Suppose we have $N$ distribution $p^\theta_i$ with $i \in \{1, \dots, N\}$, assume each distribution can be written as:

$$
p_i^\theta(x) \approx p_i(x;\beta_i) = \frac{1}{Z_i} p^\theta_{\text{base}}(x) e^{\beta_i r(x)}
$$

Suppose we have a dataset $x_j \sim p_\text{base}(x)$ with $j\in \{1,\cdots,M\}$, and we can measure $p_\text{base}(x_j)$, $p^\theta_i(x_j)$ and $r_j \coloneqq r(x_j)$ for all $i \in [N]$ and $j \in [M]$, so the empirical observed distribution as:

$$
q_{ij} \coloneqq \frac{ p^\theta_i(x_j) / p^\theta_\text{base}(x_j) } { \sum_{k=1}^M p_{i}^\theta (x_k) / p^\theta_\text{base}(x_k) }
$$

And the probability under our theoretical model is:

$$
\begin{align}

p_{ij}(\beta_i) 
&=\frac{e^{\beta_i r_j}}{\sum_{k=1}^M e^{\beta_i r_k}} 

\end{align}
$$

and we fit the $\beta$ such that the theoretical model well “explains” the observed data for all $i\in N$:

$$
\begin{align} 
&\argmin_{\beta_i} D_{KL} \left(q_i(x), p_i(x;\beta_i) \right) \\

= &\argmin_{\beta_i} \sum_{j=1}^M q_{ij} \log \left( \frac{q_{ij}}{ e^{\beta_i r_j} / \sum^M_{k=1}  e^{ \beta_i r_k}} \right) \\

= &\argmax_{\beta_i}  \left[ \beta_i \sum_{j=1}^M q_{ij} r_j - \log \sum_{j=1}^M  e^{\beta_i r_j} \right]

\end{align}
$$

and the update rule is:

$$
\beta^{t+1}_i = \beta_i^t + \frac{ \overbrace{\sum_{j=1}^M q_{ij} r_j}^{\mathbb{E}_{q_i}[r]} - \overbrace{\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j}^{ \mathbb{E}_{p_i}[r]} }{ \underbrace{ \sum_{j=1}^M p_i(x_j;\beta_i^t) r_j^2 - (\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j )^2 }_{\text{Var}_{p_i}[r]}}
$$

where:

$$
p_i(x_j;\beta^t_i) = \frac{ \exp(\beta^t_i r_j)}{\sum_{k=1}^M  \exp(\beta^t_i r_k)}
$$

Now we have $\beta_i$ for $i \in N$, it represent which $i$-th distribution is more correlate to the high-reward samples than the base distribution. We select the best $i^* = \argmax_{i \in N} \beta_i$ distribution for the next step.


## N (Distribution Optimize)

Suppose we have a labelled dataset, denote the samples as $x_j \sim p_\text{base}^\theta(x)$ with reward as $r_j \coloneqq r(x_j)$ for $j\in \{1,\cdots,M\}$. The target distribution is:

$$
p_\text{target}(x;\beta) = \frac{1}{Z_\beta} p^\theta_{\text{base}}(x) \exp(\beta r(x))
$$

Suppose we have a mixture of distributions:

$$
\begin{align}

\pi(x;w) &= \frac{1}{Z_w} \prod^N_{i=1} p^\theta_i(x)^{w_i} \\

&= \frac{1}{Z_w} p^\theta_{\text{base}}(x) \exp(w^\top \phi(x))

\end{align}
$$

where $\sum_{i=1}^Nw_i = 1$.

Let $\pi(x;w) = p_\text{target}(x;\beta)$, and we rewrite:

$$
\beta r(x) - \log Z_\beta = w^\top \phi(x) - \log Z_w
$$

where $\phi_i(x) = \log \frac{p_i^\theta(x)}{p_\text{base}^\theta(x)}$. Substitute each $x_j$ gives system of $M$ equations: 

$$
\begin{cases}

\beta r(x_1) &= w^\top \phi_j(x_1) + c \\
\beta r(x_2) &= w^\top \phi_j(x_2)  + c \\

& \; \vdots \\

\beta r(x_M) &= w^\top \phi_j(x_M)  + c 

\end{cases}
$$

notice the $c=\log Z_\beta - \log Z_w$ can cancels off, and we can write it in vector form:

$$
\begin{align}

\beta \tilde{r} &= \tilde{\Phi} w

\end{align}
$$

where $\tilde{\Phi} = [\tilde{\phi}_1, \cdots, \tilde{\phi}_M]^\top \in \mathbb{R}^{M \times N}$ is centering with $\tilde{\phi}_j \coloneqq \phi(x_j) - \frac{1}{M}\sum_{k=1}^M \phi(x_k)$, and $\tilde{r} = [\tilde{r}_1,\cdots,\tilde{r}_M]^\top \in \mathbb{R}^M$ is centering with $\tilde{r}_j = r(x_j) - \frac{1}{M} \sum_{k=1}^M r(x_k)$ . We express the $\beta$ as a function of $w$ using pseudo inverse:

$$
\begin{align}

\beta(w) &= \tilde{r}^+ \tilde{\Phi} w \\

\beta(w) &= \underbrace{ \frac{ \tilde{r}^\top \tilde{\Phi} }{ \| \tilde{r} \|^2_2}}_{\coloneqq g^\top } w

\end{align}
$$

Thus the optimization problem is:

$$
\max_{w} g^\top w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1, \| w \|^2_2 \le N
$$

this is typical linear regression, the closed-form solution is:

$$
w^* = \frac{1}{N} \mathbf{1} + \sqrt{N - \frac{1}{N}} \frac{\tilde{g}}{ \| \tilde{g} \|_2} 
$$

where it is centering with $\tilde{g} \coloneqq g - \bar{g} \mathbf{1}$  with $\bar{g} \coloneqq \frac{1}{N} \mathbf{1}^\top g$ .

Suppose the $N$ is too large, it will be expensive, we can matigate it by selecting only $n \ll N$ experts:

$$
\begin{align}
\max_{w} \quad & g^\top w \\

\text{s.t.} \quad & \mathbf{1}^\top w = 1, \\

& \| w \|^2_2 \le n \\
& \| w \|_0 \le n

\end{align}

$$

The solution have only subset $S \subset \{1, \cdots, N \}$ with $|S|=n$ of weights being activated, and the remaining be all zeros, and it is in the same form:

$$

w^*_i = 
\begin{cases}
\frac{1}{n}  + \sqrt{n - \frac{1}{n}} \frac{g_i - \bar{g}_S}{ \| \tilde{g}_S \|_2}, \quad & i \in S \\
0, \quad & i \notin S
\end{cases}

$$

where $\tilde{g}_S \coloneqq g_S - \bar{g}_S \mathbf{1}$  and  $\bar{g}_S \coloneqq \frac{1}{n} \sum_{i \in S } g_i$. 

We reordering the index such that:

$$
g_{(1)} \le g_{(2)} \cdots \le g_{(N)}
$$

and define the “head-and-tail” subset:

$$
S_k = \underbrace{\{(1),\cdots, (k)\} }_\text{head} \cup \underbrace{\{(N-n+k+1), \cdots, (N)\}}_\text{tail}
$$

and the optimal split $k^*$ is:

$$
k^* = \argmax_{k \in \{0,\cdots,n\} } \left[ \bar{g}_{S_k} + \sqrt{n - \frac{1}{n}} \| \tilde{g}_{S_k} \|_2 \right]
$$

and thus the optimal subset is $S^* = S_{k^*}$.