# Product of distributions
Suppose we have $N$ distribution $p_i$ with $i \in \{1, \dots, N\}$, assume each distribution can be written as:

$$
p_i(x;\beta_i) = \frac{1}{Z_i} p_{\text{base}}(x) e^{\beta_i r(x)}
$$

Assume we have $M$ samples $x_j \sim p_\text{base}$ with reward $r_j=r(x_j)$ for $j \in \{1,\cdots,M\}$, and the empirical base distribution is $p_\text{base}(x) = \frac{1}{M}$ for $x\in \mathcal{D}$.

We measure the $p_{ij} \coloneqq p^\theta_i(x_j)$ and $r(x_j)$ for all $i \in [N]$ and $j \in [M]$, and compute the empirical distribution as:

$$
q_{ij} \coloneqq \frac{ p_{ij} } { \sum_{k=1}^M p_{ik} (x_k) }
$$

and we can estimate the parameters $\theta = \{\beta_1,\cdots, \beta_N\}$ by the KL divergences:

$$
\begin{align} 
&\argmin_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N D_{KL} \left(q_i(x), p_i(x;\beta_i) \right) \\

= &\argmin_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N \sum_{j=1}^M q_{ij} \log \left( \frac{q_{ij}}{ e^{\beta_i r_j} / \sum^M_{k=1} e^{\beta_i r_k}} \right) \\

= &\argmax_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N  \left[ \beta_i \sum_{j=1}^M q_{ij} r_j - \log \sum_{j=1}^M e^{\beta_i r_j} \right]

\end{align}
$$

and the update rule is:

$$
\beta^{t+1}_i = \beta_i^t + \frac{\sum_{j=1}^M q_{ij} r_j - \sum_{j=1}^M p_i(x_j;\beta_i^t) r_j }{\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j^2 - (\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j )^2}
$$

where:

$$
p_i(x_j;\beta^t_i) = \frac{\exp(\beta^t_i r_j)}{\sum_{k=1}^M \exp(\beta^t_i r_k)}
$$

Now we have $\mathcal{B}^* = \{ \beta^*_1,\cdots, \beta^*_N \}$, and suppose we mix these distributions:

$$
\pi_W(x) = \frac{1}{Z_W} \prod_{i=1}^N p_i(x)^{w_i}
$$

with $\sum_{i=1}^N w_i = 1$ then the expected reward is:

$$
\begin{align} r_W &= \mathbb{E}_{x \sim \pi_W}[r(x)] \\ &= \frac{1}{Z_W} \int r(x) \prod_{i=1}^N p_i(x)^{w_i} \, dx \nonumber \\ &= \frac{1}{Z_W} \int r(x) \prod_{i=1}^N \left( \frac{1}{Z_i} p_{\text{base}}(x) \exp\left( \beta_i r(x) \right) \right)^{w_i} dx \nonumber \\ &= \frac{\prod_i Z_i^{-w_i}}{Z_W} \int r(x) p_{\text{base}}(x)^{\sum_i w_i} \exp\left( \sum_{i=1}^N w_i \beta_i r(x) \right) dx \nonumber  \\

&= \frac{1}{Z(\beta_W)} \int r(x)  p_{\text{base}}(x) \exp\left( \beta_W r(x) \right)dx \nonumber

\end{align}
$$

where $\beta_W = \sum_i w_i \beta_i$ and $Z(\beta_W) = Z_W \prod_i Z_i^{w_i} = \int p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx$ . Notice:

$$
\begin{align}

\frac{\partial}{\partial \beta_W} \log \big( Z(\beta_W) \big) &= \frac{ Z'(\beta_W)}{Z(\beta_W)} \\

&= \frac{1}{Z(\beta_W)} \frac{\partial}{\partial \beta_W} \int p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx \\

&= \frac{1}{Z(\beta_W)} \int p_{\text{base}}(x) \frac{\partial}{\partial \beta_W}  \exp\left( \beta_W r(x) \right) dx \\

&= \frac{1}{Z(\beta_W)} \int r(x) p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx \\

&= r_W

\end{align}
$$

and:

$$
\begin{align}
\frac{\partial^2}{\partial \beta_W^2} \log \big( Z(\beta_W) \big)

&= \frac{\partial}{\partial \beta_W} \left( \frac{ Z'(\beta_W)}{Z(\beta_W)} \right) \\

&= \frac{Z(\beta_W)Z''(\beta_W) - Z'(\beta_W)Z'(\beta_W)}{Z(\beta_W)^2} \\
&= \frac{Z''(\beta_W)}{Z(\beta_W)} - \left( \frac{Z'(\beta_W)}{Z(\beta_W)} \right)^2 \\

&= \mathbb{E}_{x \sim \pi_W}[r(x)^2] - \mathbb{E}_{x \sim \pi_W}[r(x)]^2 \\

&= \text{Var}_{x \sim \pi_W}[r(x)] \\

& > 0

\end{align}
$$

This prove that the expected reward $r_W$ is monotonically increasing with respect to $\beta_W$, so

$$
\argmax_W r_W = \argmax_W \beta_W
$$

We impose constraints so that $W$ has closed-form solution:

$$
\argmax_W  \beta_W \\
\text{subject to} \sum_i w_i=1 \text{ and } \frac{1}{N} \sum_i w_i^2 = 1,
$$

the solution is:

$$
w^*_i = \frac{1}{N} + \sqrt{N - \frac{1}{N}} \frac{\beta_i -\bar{\beta}}{\sqrt{ \sum_j (\beta_j -\bar{\beta})^2 }}
$$

where $\bar{\beta} = \frac{1}{N}\sum_j \beta_j$.

# Sum of distributions
Suppose we have $N$ distribution $p_i$ with $i \in \{1, \dots, N\}$, and we have $M$ samples $x_j$ with reward $r_j=r(x_j)$ for $j \in \{1,\cdots,M\}$.

We measure the $p_{ij} \coloneqq p^\theta_i(x_j)$ and $r(x_j)$ for all $i \in [N]$ and $j \in [M]$, and compute the empirical distribution as:

$$
q_{ij} \coloneqq \frac{ p_{ij} } { \sum_{k=1}^M p_{ik} (x_k) }
$$

and the expected reward for $i$-th distribution is:

$$
\hat{r}_i = \sum_{j=1}^M q_{ij} r_j
$$

Suppose we mix these distributions:

$$
\pi_W(x) = \sum_{i=1}^N w_i p_i(x)
$$

with $\sum_{i=1}^N w_i = 1$ then the expected reward is:

$$
\begin{align} r_W &= \mathbb{E}_{x \sim \pi_W}[r(x)] \\

&= \int r(x) \pi_W(x) dx \nonumber \\

&= \int r(x) \sum_{i=1}^N w_i p_i(x) dx \nonumber \\

&= \sum_{i=1}^N w_i \mathbb{E}_{x \sim p_i(x)} [r(x)] \nonumber \\

\implies \hat{r}_W &= \sum_{i=1}^N w_i \hat{r}_i

\end{align}
$$

We impose constraints so that $W$ has closed-form solution:

$$
\argmax_W  \hat{r}_W \\
\text{subject to} \sum_i w_i=1 \text{ and } \frac{1}{N} \sum_i w_i^2 = 1,
$$

the solution is:

$$
w^*_i = \frac{1}{N} + \sqrt{N - \frac{1}{N}} \frac{\hat{r}_i -\bar{r}}{\sqrt{ \sum_j (\hat{r}_j -\bar{r})^2 }}
$$

where $\bar{r} = \frac{1}{N}\sum_j \hat{r}_j$.

# Product of distributions (with base)

Suppose we have $M$ samples $x_j \sim p^\theta_\text{base}$, and we can measure $p^\theta_\text{base}(x_j)$ and reward $r_j=r(x_j)$ for $j \in \{1,\cdots,M\}$. 

We construct $p^\theta_i$ for $i \in \{1, \dots, N\}$, and <u>assume</u> each can be written as:

$$
p^\theta_i \approx p_i(x;\beta_i) = \frac{1}{Z_i} p_{\text{base}}(x) e^{\beta_i r(x)}
$$

denotes $p_{0j} \coloneqq p_\text{base}^\theta(x_j)$, and we measure the $p_{ij} \coloneqq p^\theta_i(x_j)$ and $r(x_j)$ for all $i \in \{ 0, \cdots, N \}$ and $j \in \{0,\cdots,M \}$, and compute the empirical distribution as:

$$
q_{ij} \coloneqq \frac{ p_{ij} / p_{0j} } { \sum_{k=1}^M p_{ik} / p_{0k} }
$$

and we can estimate the parameters $\theta = \{\beta_1,\cdots, \beta_N\}$ by the KL divergences:

$$
\begin{align} 
&\argmin_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N D_{KL} \left(q_i(x), p_i(x;\beta_i) \right) \\

= &\argmin_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N \sum_{j=1}^M q_{ij} \log \left( \frac{q_{ij}}{ e^{\beta_i r_j} / \sum^M_{k=1} e^{ \beta_i r_k}} \right) \\

= &\argmax_{\beta_1,\cdots, \beta_N} \sum_{i=1}^N  \left[ \beta_i \sum_{j=1}^M q_{ij} r_j - \log \sum_{j=1}^M e^{\beta_i r_j} \right]

\end{align}
$$

and the update rule is:

$$
\beta^{t+1}_i = \beta_i^t + \frac{\sum_{j=1}^M q_{ij} r_j - \sum_{j=1}^M p_i(x_j;\beta_i^t) r_j }{\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j^2 - (\sum_{j=1}^M p_i(x_j;\beta_i^t) r_j )^2}
$$

where:

$$
p_i(x_j;\beta^t_i) = \frac{\exp(\beta^t_i r_j)}{\sum_{k=1}^M \exp(\beta^t_i r_k)}
$$

Now we have $\beta_0=0$ and $\mathcal{B}^* = \{ \beta^*_1,\cdots, \beta^*_N \}$, and suppose we mix these distributions:

$$
\pi_W(x) = \frac{1}{Z_W} \prod_{i=0}^N p^\theta_i(x)^{w_i}
$$

with $\sum_{i=0}^N w_i = 1$ then the expected reward is:

$$
\begin{align} r_W &= \mathbb{E}_{x \sim \pi_W}[r(x)] \\ &= \frac{1}{Z_W} \int r(x) \prod_{i=0}^N p_i(x)^{w_i} \, dx \nonumber \\ &= \frac{1}{Z_W} \int r(x) \prod_{i=0}^N \left( \frac{1}{Z_i} p_{\text{base}}(x) \exp\left( \beta_i r(x) \right) \right)^{w_i} dx \nonumber \\ &= \frac{\prod_i Z_i^{-w_i}}{Z_W} \int r(x) p_{\text{base}}(x)^{\sum_i w_i} \exp\left( \sum_{i=0}^N w_i \beta_i r(x) \right) dx \nonumber  \\

&= \frac{1}{Z(\beta_W)} \int r(x)  p_{\text{base}}(x) \exp\left( \beta_W r(x) \right)dx \nonumber

\end{align}
$$

where $\beta_W = \sum_i w_i \beta_i$ and $Z(\beta_W) = Z_W \prod_i Z_i^{w_i} = \int p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx$ . Notice:

$$
\begin{align}

\frac{\partial}{\partial \beta_W} \log \big( Z(\beta_W) \big) &= \frac{ Z'(\beta_W)}{Z(\beta_W)} \\

&= \frac{1}{Z(\beta_W)} \frac{\partial}{\partial \beta_W} \int p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx \\

&= \frac{1}{Z(\beta_W)} \int p_{\text{base}}(x) \frac{\partial}{\partial \beta_W}  \exp\left( \beta_W r(x) \right) dx \\

&= \frac{1}{Z(\beta_W)} \int r(x) p_{\text{base}}(x) \exp\left( \beta_W r(x) \right) dx \\

&= r_W

\end{align}
$$

and:

$$
\begin{align}
\frac{\partial^2}{\partial \beta_W^2} \log \big( Z(\beta_W) \big)

&= \frac{\partial}{\partial \beta_W} \left( \frac{ Z'(\beta_W)}{Z(\beta_W)} \right) \\

&= \frac{Z(\beta_W)Z''(\beta_W) - Z'(\beta_W)Z'(\beta_W)}{Z(\beta_W)^2} \\
&= \frac{Z''(\beta_W)}{Z(\beta_W)} - \left( \frac{Z'(\beta_W)}{Z(\beta_W)} \right)^2 \\

&= \mathbb{E}_{x \sim \pi_W}[r(x)^2] - \mathbb{E}_{x \sim \pi_W}[r(x)]^2 \\

&= \text{Var}_{x \sim \pi_W}[r(x)] \\

& > 0

\end{align}
$$

This prove that the expected reward $r_W$ is monotonically increasing with respect to $\beta_W$, so

$$
\argmax_W r_W = \argmax_W \beta_W
$$

We impose constraints so that $W$ has closed-form solution:

$$
\argmax_W  \sum_i w_i \beta_i \\
\text{subject to} \sum_i w_i=1 \text{ and } \frac{1}{N+1} \sum_i w_i^2 = 1,
$$

the solution is:

$$
w^*_i = \frac{1}{N+1} +  \sqrt{N+1 - \frac{1}{N+1}} \frac{\beta^*_i -\bar{\beta^*}}{\sqrt{ \sum_{j=0}^N (\beta^*_j -\bar{\beta^*})^2 }}
$$

where $\bar{\beta} = \frac{1}{N+1}\sum_{j=0}^N \beta_j$.