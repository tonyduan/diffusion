# Diffusion Models From Scratch

March 2023.

---

Here, we'll cover the derivations from scratch to provide a rigorous understanding of the core ideas behind diffusion. What assumptions are we making? What properties arise as a result?

A reference [[codebase]](https://www.github.com/tonyduan/diffusion) is written from scratch, which provides minimalist re-production of the MNIST example below. It clocks in at well under 500 LOC.

Any errors are mine.

![MNIST](examples/ex_mnist_crop.png)

(Left: MNIST groundtruth. Right: MNIST sampling starting from random noise).

#### Contents

1. Structure
2. The Backward Model
3. The Forward Model
4. Training
5. Accelerated Sampling

---

#### I. Structure

A diffusion model hypothesizes an observed variable $\mathbf{x}_0$ and latent variables $\mathbf{x}_1,\dots,\mathbf{x}_T$ arranged in the following graphical model.

$$
\begin{equation*}
\mathbf{x}_T \rightarrow \dots \rightarrow \mathbf{x}_{t} \rightarrow \mathbf{x}_{t-1} \rightarrow \dots \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_0
\end{equation*}
$$

We assume the following forward and backward models respectively.

$$
\begin{align*}
p_\theta(\mathbf{x}_{0:T}) & = p(\mathbf{x}_T)\prod_t p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t), \quad p(\mathbf{x}_T) = N(\mathbf{0}, \mathbf{I})\\
q(\mathbf{x}_{1:T}|\mathbf{x}_0) & = \prod_t q(\mathbf{x}_t|\mathbf{x}_{t-1})
\end{align*}
$$

In later sections we'll define specific distributional assumptions. But for now it suffices to note the following.

**Remark**. The backward model $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ is *fixed*. All learnable parameters lie in the forward model $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$.

---

As with any latent-variable model, the training objective is to maximize marginal log-likelihood.

$$
\begin{align*}
\log p_\theta(\mathbf{x}_0) & = \int_{\mathbf{x}_{1:T}}\log p_\theta(\mathbf{x}_0,\mathbf{x}_{1:T})d\mathbf{x}_{1:T}
\end{align*}
$$

But this is intractable to compute due to the un-observed variables. Instead we'll maximize the Evidence Lower Bound (ELBO) which arises from Jensen's inequality.

$$
\begin{align*}
\log p_\theta(\mathbf{x}_0) & = \log \int_{\mathbf{x}_{1:T}} p_\theta(\mathbf{x}_0,\mathbf{x}_{1:T})d\mathbf{x}_{1:T}\\
& = \log \int_{\mathbf{x}_{1:T}}q(\mathbf{x}_{1:T}|\mathbf{x}_0)\frac{p_\theta(\mathbf{x}_0,\mathbf{x}_{1:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}d\mathbf{x}_{1:T}\\
& \geq \int_{\mathbf{x}_{1:T}}q(\mathbf{x}_{1:T}|\mathbf{x}_0)\log \frac{p_\theta(\mathbf{x}_0,\mathbf{x}_{1:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}d\mathbf{x}_{1:T} \triangleq L_\theta(\mathbf{x}_0)
\end{align*}
$$

Why is this a reasonable quantity to optimize?

**Result**. The gap between the marginal log-likelihood and the ELBO is exactly the KL divergence between the true forward model $p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)$ (which is intractable) and the hypothesized backward model $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$. So by maximizing the ELBO we're optimizing how well we approximate the hypothesized backward model.

**Proof**. Let's compute the gap between the marginal likelihood and the ELBO.
$$
\begin{align*}
\log p_\theta(\mathbf{x_0}) - L_{\theta}(\mathbf{x_0})
& = \log \int_{\mathbf{x}_{1:T}} p_\theta(\mathbf{x}_0,\mathbf{x}_{1:T}) d\mathbf{x}_{1:T} -  \int_{\mathbf{x}_{1:T}} q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \frac{p_\theta(\mathbf{x}_0)p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}d\mathbf{x}_{1:T}\\
& = \log \int_{\mathbf{x}_{1:T}} p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T} + \log p_\theta(\mathbf{x}_0) -\\& \quad\quad\quad  \int_{\mathbf{x}_{1:T}} q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \frac{p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0)}d\mathbf{x}_{1:T} - \log p_\theta(\mathbf{x}_0)\\
& = \log \int_{\mathbf{x}_{1:T}} p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T}  -  \int_{\mathbf{x}_{1:T}} q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \frac{p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0)}d\mathbf{x}_{1:T} \\
& = D_\mathrm{KL}(\ q_\phi(\mathbf{x}_{1:T}|\mathbf{x}_0)\ \|\ p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)\ ) \geq 0
\end{align*}
$$

On the last line above we invoked the non-negativity of KL divergence.

---

**Result**. From the structure of the graphical model assumed, we will be able to simplify the ELBO into a convenient form.
$$
\begin{align*}
L_\theta(\mathbf{x}_0) & = \int_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \prod_{t=1}^Tq(\mathbf{x}_t|\mathbf{x}_{t-1}) \log \frac{p(x_T)\prod_{t} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{\prod_{t} q(\mathbf{x}_t|\mathbf{x}_{t-1})}d\mathbf{x}_{1:T}\\
& = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p(\mathbf{x}_T) + \sum_{t} \log p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) -\sum_{t}\log q(\mathbf{x}_t|\mathbf{x}_{t-1}) \right]\\
& = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p(\mathbf{x}_T) + \sum_{t>1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} + \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)}\right]\\
& = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p(\mathbf{x}_T) + \sum_{t>1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)}\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} + \log \frac{p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)}\right]\\
& = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} + \sum_{t>1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)} + \log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]\\
& = \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1) - \sum_{t>1} D_\mathrm{KL}(\ q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\ \|\ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\ )\right] - D_\mathrm{KL}(\ q(\mathbf{x}_T|\mathbf{x}_0)\ \|\ p(\mathbf{x}_T)\ ) \\
& \propto \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1) - \sum_{t>1} D_\mathrm{KL}(\ q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\ \|\ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\ ) \right]\\
& = \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}\left[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]}_{L_1} - \sum_{t>1} \underbrace{\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left[D_\mathrm{KL}(\ q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\ \|\ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\ )\right]}_{L_t}
\end{align*}
$$

On the fourth line above we applied

1. Markov property $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)$.
2. Bayes' Rule $q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0) = \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}$.

On the seventh line we used the fact that there are no learnable parameters in the expression which was removed, $D_\mathrm{KL}(\ q(\mathbf{x}_T|\mathbf{x}_0)\ \|\ p(\mathbf{x}_T)\ )$.

The last line is important because it makes the expectations explicit (for whatever reason most references I find online omit this detail, which I find crucial for understanding). It makes the following easy to see.

**Result**. Each individual term in the ELBO relies only on *factorized* conditional distributions $q(\mathbf{x}_t|\mathbf{x}_0)$ instead of the joint $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$.

This is important because we'll later optimize the ELBO via Monte Carlo gradient estimates over these expectations. That is, the algorithm will look something like the following.

```python
def compute_L_t(x_0, t):
    monte_carlo_x_t = sample q(x_t | x_0)
    true_distn = get_gt(monte_carlo_x_t, x_0)
    pred_distn = get_pred(monte_carlo_x_t)  # gradient flows into model here
    loss = kl_div(true_distn, pred_distn)
    return loss
```

This is similar to the training procedure for a variational auto-encoder. But because there are no learnable parameters in $q(\mathbf{x}_t|\mathbf{x}_0)$, the re-parameterization trick (or a similar technique such as log-derivative trick) is not even necessary. A simple Monte Carlo sample suffices.

**Remark**. Diffusion models do not admit tractable log-likelihood evaluation due to the inability to marginalize over latent variables $\mathbf{x}_{1:T}$.

This puts it in the same class as variational auto-encoders. In contrast, autoregressive models and normalizing flows admit tractable log-likelihoods. For comparison to such likelihood-based generative models, Ho et al. report metrics in terms of the ELBO.

---

#### II. The Backward Model

As we will see, how we define $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ will be the key to deriving several properties.

**Define**. The backward process is the following.

$$
\begin{align*}
q(\mathbf{x}_t|\mathbf{x}_{t-1}) & = N(\sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})\\
\alpha_t & \triangleq 1- \beta_t\\
\bar\alpha_t & \triangleq \prod_{s=1}^t \alpha_s
\end{align*}
$$

To guide intuition, a typical choice is linearly increasing $\beta_1=10^{-4} \rightarrow\beta_T=2 \times 10^{-2}$. (We'll revisit this choice of hyper-parameter later on).

The interpretation of $q(\mathbf{x}_{t}|\mathbf{x}_{t-1})$ is that we hypothesize isotropic latent features where $\mathbf{x}_t$ is a noisier version of $\mathbf{x}_{t-1}$.

It is worth commenting upon the use of notation. Initially I found it confusing to use both $\alpha_t$ and $\beta_t$, since they're not independent variables and one is easily derived from the other. But the use of both simplifies the math quite a bit, even though it's redundant. So I'll keep the notation, which is consistent with Ho et al. (2020). It is worth noting that some papers such as Song et al. (2021) exclusively use $\bar\alpha_t$ and do not use $\beta_t$ at all.

**Key Result**. Since $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ has no learnable parameters, we can derive the following closed form representations.
$$
\begin{align*}
q(\mathbf{x}_t| \mathbf{x}_{0}) & = N(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar\alpha_t) \mathbf{I}),\\
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) & = N\left(\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} \mathbf{x}_0+ \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\mathbf{I}\right).
\end{align*}
$$

This is the most important result on this page.

**Proof**.

Observe that we can re-write $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ as the following.

$$
\begin{align*}
\mathbf{x}_t |\mathbf{x}_{t-1}= \sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{1-\alpha_t} \boldsymbol\epsilon_t,\quad \boldsymbol\epsilon_t \sim N(\mathbf{0}, \mathbf{I})\\
\end{align*}
$$

Let's unroll one step of the recursive process.

$$
\begin{align*}
\mathbf{x}_{t}|\mathbf{x}_{t-2} & = \sqrt{\alpha_t}(\mathbf{x}_{t-1}|\mathbf{x}_{t-2}) + \sqrt{1-\alpha_t}\boldsymbol\epsilon_t\\
& = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}} \boldsymbol\epsilon_{t-1}) + \sqrt{1-\alpha_t}\boldsymbol\epsilon_t\\
& = \sqrt{\alpha_t \alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol\epsilon_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon_t}
\end{align*}
$$

Recall that linear combination of Gaussians yields another Gaussian.

$$
\begin{equation*}
N(\boldsymbol\mu_1,\boldsymbol\Sigma_1)+N(\boldsymbol{\mu}, \boldsymbol{\Sigma}_2)=N(\boldsymbol\mu_1 + \boldsymbol\mu_2, \boldsymbol\Sigma_1 + \boldsymbol\Sigma_2)
\end{equation*}
$$

Then it follows that

$$
\begin{align*}
\mathbf{x}_t|\mathbf{x}_{t-2} & = \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\boldsymbol\epsilon_t.
\end{align*}
$$

It's straightforward to extend this process to form an inductive proof.

$$
\begin{align*}
\mathbf{x}_t |\mathbf{x}_0& = \left(\prod_{s=1}^t\sqrt{\alpha_s}\right)\mathbf{x}_0 + \sqrt{1-\prod_{s=1}^t\alpha_s} \boldsymbol\epsilon
\end{align*}
$$

This gives us the result for $q(\mathbf{x}_t | \mathbf{x}_0)$.

Next we need to tackle $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$. It will help here to recall the density for a multivariate normal parameterized $N(\boldsymbol\mu,\sigma^2\mathbf{I})$,

$$
\begin{align*}
p(\mathbf{z};\boldsymbol\mu,\sigma^2) & \propto \exp\left(-\frac{1}{2}\left(\frac{\|\mathbf{z}-\boldsymbol\mu\|^2}{\sigma^2}\right)\right) \\
& \propto\exp\left(-\frac{1}{2}\left(\frac{\|\mathbf{z}\|^2}{\sigma^2}- \frac{2\mathbf{z}^\top\boldsymbol\mu}{\sigma^2}\right)\right)
\end{align*}
$$

We then use Bayes' Rule and a bit of algebra to end up with another Gaussian. Note that we only care about terms that contain $\mathbf{x}_{t-1}$ and therefore ignore the other terms to go from the second to third lines below.

$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_{t,}\mathbf{x}_0) & = \frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_{0})}\\
& \propto \exp\left(-\frac{1}{2}\left(\frac{\|\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0\|^2}{1-\bar{\alpha_{t-1}}} - \frac{\|\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0\|^2}{1-\bar\alpha_{t}}\right)\right)\\
& \propto \exp\left(-\frac{1}{2}\left(\frac{-2\sqrt{\alpha_t}\mathbf{x}_t^\top\mathbf{x}_{t-1}+\alpha_t\|\mathbf{x}_{t-1}\|^2}{\beta_t} + \frac{\|\mathbf{x}_{t-1}\|^2 -2\sqrt{\bar\alpha_{t-1}}\mathbf{x}_{t-1}^\top\mathbf{x}_0}{1-\bar\alpha_{t-1}}\right)\right)\\
& = \exp\left(-\frac{1}{2}\left(\underbrace{\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)}_{1/\sigma^2}\|\mathbf{x}_{t-1}\|^2 - \underbrace{\left(\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0\right)^\top}_{2\boldsymbol{\mu}/\sigma^2}\mathbf{x}_{t-1}\right)\right)
\end{align*}
$$

If we solve for the parameters $\boldsymbol\mu, \sigma^2$ above we can find

$$
\begin{align*}
\sigma^2 & = \left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1} = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\\
\boldsymbol\mu & = \left(\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0\right)\sigma^2 = \frac{\sqrt\alpha_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t + \frac{\sqrt{\bar\alpha_t\beta_t}}{1-\bar\alpha_t}\mathbf{x}_0.
\end{align*}
$$

This completes the proof.

**Result**. Importantly, we will later use the fact that we can re-parameterize $q(\mathbf{x}_{t}|\mathbf{x}_0)$ by writing it in the following form.
$$
\begin{align*}
\mathbf{x}_t |\mathbf{x}_0&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon, &\boldsymbol\epsilon \sim N(\mathbf{0}, \mathbf{I})\\
\mathbf{x}_0 & = \frac{1}{\sqrt{\bar\alpha_t}}\left((\mathbf{x}_t |\mathbf{x}_0)- \sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon\right),&\boldsymbol\epsilon \sim N(\mathbf{0}, \mathbf{I})
\end{align*}
$$

This will allow us to use the re-parameterization trick (Kingma and Welling 2014). We can update the training pseudocode accordingly:
```python
def compute_L_t(x_0, t):
    eps = sample N(0, I)
    monte_carlo_x_t = sqrt_bar_alpha[t] + sqrt_one_minus_bar_alpha[t] * eps
    true_distn = get_gt(monte_carlo_x_t, x_0)
    pred_distn = get_pred(monte_carlo_x_t)  # gradient flows into model here
    loss = kl_div(true_distn, pred_distn)
    return loss
```

---

#### Choice of Noise Schedule

Earlier above we introduced a linear schedule for $\beta_t$. An alternative way to set this hyper-parameter is via a *cosine* noise schedule, defined as the following (Nichol and Dhariwal 2021).
$$
\begin{align*}
\bar\alpha_t & = \frac{f(t)}{f(0)} & f(t) & = \cos\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right)^2 & \beta_t&=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}},
\end{align*}
$$
where $s=0.008$ is a hyper-parameter.

The motivation is as follows. Recall that $q(\mathbf{x}_t|\mathbf{x}_0)\sim N(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$. We can therefore interpret $1-\bar\alpha_t$ as the amount of noise we'll be adding to $\mathbf{x}_0$ to receive a Monte Carlo sample of $\mathbf{x}_t|\mathbf{x}_0$. It turns out the linear schedule results in $1-\bar\alpha_t \approx 1$ for most samples when $t$ is large, which is undesirable because it's too noisy and difficult to learn. The cosine schedule, on the other hand, results in a more linear interpolation for $\bar\alpha_t$ despite being non-linear in $\beta_t$.

---

#### III. The Forward Model

**Define**. For the forward process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ we'll use one of the following.
$$
\begin{align*}
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }1)} & = N\left(\ \underbrace{f_\theta(\mathbf{x}_t,t)}_{\mu_\theta(\mathbf{x}_t, t)}, \underbrace{\sigma_t^2 \mathbf{I}}_{\Sigma_\theta(\mathbf{x}_t,t)}\right)\\
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }2)} & = N\left(\ \underbrace{\mathbb{E}\left[q(\mathbf{x}_{t-1}|\mathbf{x}_t,f_\theta(\mathbf{x}_t, t))\right]}_{\mu_\theta(\mathbf{x}_t, t)}, \underbrace{\sigma_t^2 \mathbf{I}}_{\Sigma_\theta(\mathbf{x}_t,t)}\right)\\
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }3)} & = N\left(\ \underbrace{\mathbb{E}\left[q\left(\mathbf{x}_{t-1}|\mathbf{x}_t,\frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(\mathbf{x}_t, t)\right)\right)\right]}_{\mu_\theta(\mathbf{x}_t, t)}, \underbrace{\sigma_t^2 \mathbf{I}}_{\Sigma_\theta(\mathbf{x}_t,t)})\right)
\end{align*}
$$

We will soon motivate and explain the differences between the three options.

For now, it suffices to acknowledge that they differ *only* in how they parameterize means. That is, variances are *fixed* and not learned. We'll come back to the question of variance later -- but for now, let's dive into what's going on in the mean parameterization.

---

#### Mean Parameterization

Recall that in the backward model, our key result was that

$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) & = N\left(\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} \mathbf{x}_0+ \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t} \beta_t\mathbf{I}\right).
\end{align*}
$$

Next, recall that the KL divergence between $q_1 = N(\boldsymbol\mu_1,\boldsymbol\Sigma_1)$ and $q_2= N(\boldsymbol\mu_2,\boldsymbol\Sigma_2)$ in $d$ dimensions has a closed form,

$$
\begin{align*}
D_\mathrm{KL}(\ q_1\ \|\ q_2\ ) &= \frac{1}{2}\left(\log \frac{|\boldsymbol\Sigma_2|}{|\boldsymbol\Sigma_1|} - d+\mathrm{tr}(\boldsymbol\Sigma^{-1}_2\boldsymbol\Sigma_1) + (\boldsymbol\mu_2-\boldsymbol\mu_1)^\top \boldsymbol\Sigma_2^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)\right).
\end{align*}
$$

Because we fixed the variance of $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ (i.e. it's not learnable), all but the last term in the above expression are constant with respect to $\theta$. This simplifies the math quite a bit.

**Result**. We can write the ELBO loss as the following.
$$
\begin{align*}
L_t(\mathbf{x}_0) & = \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}[D_\mathrm{KL}(\ q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)\ \|\ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)\ )]\\
& \propto\frac{1}{2\sigma_t^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left\|\underbrace{\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} \mathbf{x}_0+ \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t}_{\text{true mean}} - \underbrace{\mu_\theta(\mathbf{x}_t,t)}_\text{predicted mean}\right\|^2
\end{align*}
$$

**Option 1**. Directly learn the true mean.

We can stop here and directly learn the true mean of $\mathbf{x}_{t-1}$ as a function $f_\theta(\mathbf{x}_t,t)$.

$$
\begin{align*}
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }1)} & = N\left(\ \underbrace{f_\theta(\mathbf{x}_t,t)}_{\mu_\theta(\mathbf{x}_t, t)}, \sigma_t^2 \mathbf{I}\right)\\
L_t(\mathbf{x}_0)^\text{(option 1)} &= \frac{1}{2\sigma_t^2} \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left\|\underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 +\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t}_{\text{true mean}} - \underbrace{\mu_\theta(\mathbf{x}_t, t) }_\text{predicted mean}\right\|^2\\
 &= \frac{1}{2\sigma_t^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)} \left\|\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 +\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t - f_\theta(\mathbf{x}_t, t) \right\|^2\\
\end{align*}
$$

One note here for later on is that this parameterization is not compatible with accelerated sampling. The reason is because the regression target here depends on $\mathbf{x}_t$ (and, consequently, on $t$). Ho et al. (2020) found that this parameterization works the least well out of the options listed here. It's included for education purposes only.

**Option 2**. Learn to de-noise (Song et al. 2021).

Hypothesize the predicted mean of the forward process as the mean of the distribution $q(\mathbf{x}_{t-1}|\mathbf{x}_t,f_\theta(\mathbf{x}_0,t))$. That is, the mean of the posterior distribution where the *predicted* de-noised observation $f_\theta(\mathbf{x}_0, t)$ is substituted for the *true* de-noised observation $\mathbf{x}_0$.

$$
\begin{align*}
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }2)} & = N\left(\ \underbrace{\mathbb{E}\left[q(\mathbf{x}_{t-1}|\mathbf{x}_t,f_\theta(\mathbf{x}_t, t))\right]}_{\mu_\theta(\mathbf{x}_t, t)}, \sigma_t^2 \mathbf{I}\right)\\
L_t(\mathbf{x}_0)^\text{(option 2)} & \propto \frac{1}{2\sigma_t^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left\|\underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 +\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t}_{\text{true mean}} - \underbrace{\mu_\theta(\mathbf{x}_t, t) }_\text{predicted mean}\right\|^2\\
& = \frac{1}{2\sigma_t^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)} \left\|\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0  - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}f_\theta(\mathbf{x}_t,t)  \right\|^2\\
& = \frac{\bar{\alpha}_{t-1}\beta_t^2}{2\sigma_t^2(1-\bar{\alpha}_t)^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\|\mathbf{x}_0-f_\theta(\mathbf{x}_t,t)\|^2
\end{align*}
$$

Here, we can interpret $f_\theta(\mathbf{x}_t,t)$ as directly predicting the de-noised $\mathbf{x}_0$. With this parameterization, there are no $\mathbf{x}_t$ terms in the regression target. Instead, the regression target depends only on $\mathbf{x}_0$ and not $t$. This fact will prove useful later on when we derive the accelerated sampling process.

**Option 3**. Learn to predict noise (Ho et al. 2020).

This parameterization is similar to the previous one. But instead of predicting the true de-noised observation $\mathbf{x}_0$, we'll take advantage of how we're taking Monte Carlo samples and instead predict $\boldsymbol\epsilon$, which is the *noise that was added* to $\mathbf{x}_0$ to produce the Monte Carlo sample of $\mathbf{x}_t|\mathbf{x}_0$.

This relies on the important fact noted earlier that we're computing this loss as an expectation over $q(\mathbf{x}_{t}|\mathbf{x}_0)$. This means we can substitute the re-parameterization we derived earlier for $\mathbf{x}_t$, namely

$$
\begin{align*}
\mathbf{x}_t |\mathbf{x}_0 & = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon, & \boldsymbol\epsilon \sim N(\mathbf{0}, \mathbf{I})\\
\mathbf{x}_0 & = \frac{1}{\sqrt{\bar\alpha_t}}\left((\mathbf{x}_t |\mathbf{x}_0)- \sqrt{1-\bar{\alpha}_t}\boldsymbol\epsilon\right),& \boldsymbol\epsilon \sim N(\mathbf{0}, \mathbf{I})
\end{align*}
$$

Substituting this into the expression found for the previous option, we can write:

$$
\begin{align*}
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)^{(\text{option }3)} & = N\left(\ \underbrace{\mathbb{E}\left[q\left(\mathbf{x}_{t-1}|\mathbf{x}_t,\frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(\mathbf{x}_t, t)\right)\right)\right]}_{\mu_\theta(\mathbf{x}_t, t)}, \sigma_t^2 \mathbf{I})\right)\\
L_t(\mathbf{x}_0)^{\text{(option 3)}} & \propto \frac{1}{2\sigma_t^2} \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left\|\underbrace{\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 +\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t}_{\text{true mean}} - \underbrace{\mu_\theta(\mathbf{x}_t, t) }_\text{predicted mean}\right\|^2\\
& = \frac{\bar\alpha_{t-1}\beta_t^2}{2\sigma_t^2(1-\bar\alpha_t)^2}\mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left\| \mathbf{x}_0 - \left(\frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)\right)\right) \right\|^2\\
& = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\mathbb{E}_{q(\boldsymbol\epsilon)}\left\| \boldsymbol\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,t)\right\|^2
\end{align*}
$$

Here, $\boldsymbol\epsilon$ is the noise that was drawn to produce a Monte Carlo sample of $\mathbf{x}_t|\mathbf{x}_0$. Like predicting the de-noised obervation, choosing to predict the noise $\epsilon_\theta(\mathbf{x}_t,t)$ has no dependency on $\mathbf{x}_t$ in the regression target and so it's amenable to accelerated sampling.

To recap: forward process option 1 listed here is included for education purposes only. In our codebase and in the rest of this post we will consider only option 2 and option 3. There doesn't seem to be a consensus on which of these two options is superior in practice.

----

#### Variance Parameterization

Recall that we fixed the forward model variance to $\sigma_t^2 \mathbf{I}$. What's an appropriate choice for $\sigma^2_t$?

Ho et al. (2020) found that either of the two below options works equally well. Apparently it doesn't significantly affect sample quality which one we choose. They coincide with other quantities we're familiar with.
$$
\begin{align*}
{\sigma^2_t}^\text{(upper bound)} & = \beta_t & \implies \sigma^2_t\mathbf{I} & =\mathrm{Var}[q(\mathbf{x}_{t}|\mathbf{x}_{t-1})]\\
{\sigma^2_t}^\text{(lower bound)} & =\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t &  \implies \sigma^2_t \mathbf{I} & = \mathrm{Var}[q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)]
\end{align*}
$$

**Result**. These two options can be interpreted as upper and lower bounds on the true variance of the distribution $q(\mathbf{x}_{t-1}|\mathbf{x}_t$) under certain assumptions.

First suppose $\mathbf{x}_0 \sim \delta(\mathbf{c})$ i.e. a delta distribution at a constant. Then it's easy to see that a lower bound holds in this case.
$$
\begin{align*}
\mathrm{Var}[q(\mathbf{x}_{t-1}|\mathbf{x}_t)] \succeq \mathrm{Var}[q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)]
\end{align*}
$$
Next suppose $\mathbf{x}_0\sim N(\mathbf{c}, \tau\mathbf{I})$ for some fixed variance $\tau$, isotropically distributed around some constant. Then we have an upper bound,
$$
\begin{equation*}
\mathrm{Var}[q(\mathbf{x}_{t-1}|\mathbf{x}_t)] \preceq \mathrm{Var}[q(\mathbf{x}_{t}|\mathbf{x}_{t-1})]
\end{equation*}
$$
**Proof**. In the upper bound case, the marginal distribution of $q(\mathbf{x}_t)$ is the following.
$$
\begin{align*}
\mathbf{x}_t|\mathbf{x}_0 & = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon\\
\mathbf{x}_0 &= \sqrt\tau \boldsymbol\epsilon' + \mathbf{c}\\
\mathbf{x}_t & = \sqrt{\bar\alpha_t}(\sqrt\tau \boldsymbol\epsilon'+\mathbf{c}) + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon\\
q(\mathbf{x}_t)& = N\left(\sqrt{\bar\alpha_t}\boldsymbol{c}, \underbrace{(1+\bar\alpha_t(\tau-1))}_{\tau_t}\boldsymbol{I}\right)
\end{align*}
$$
The specific variance of the marginal distribution doesn't matter, all that matters is that $q(\mathbf{x}_t)$ is a Gaussian. With that we can apply Bayes' Rule once again to find that $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ is a Gaussian. To simplify notation we'll define $\tau_t \triangleq 1+\bar\alpha_t(\tau-1)$.
$$
\begin{align*}
q(\mathbf{x}_{t-1}|\mathbf{x}_t) & \propto q(\mathbf{x}_{t-1})q(\mathbf{x}_{t}|\mathbf{x}_{t-1})\\
& \propto \exp\left(-\frac{1}{2}\left(\frac{\|\mathbf{x}_{t-1}-\sqrt{\bar\alpha_t}\mathbf{c}\|^2}{\tau_t} + \frac{\|\mathbf{x}_{t-1}-\mathbf{x}_t\|^2}{\beta_t}\right)\right)\\
& \propto \exp\left(-\frac{1}{2}\left(\frac{\beta_t+\tau_t}{\beta_t\tau_t}\|\mathbf{x}_{t-1}\|^2+(\dots)\mathbf{x}_{t-1}^\top(\dots) \right)\right)
\end{align*}
$$
Because we're only interested in the variance of this distribution (and not the mean), we ignore the term that arises that's linear in $\mathbf{x}_{t-1}$ and hide irrelevant constants in ellipsis above. From this we can determine the variance of $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$. The next lines below show that it's upper bounded by the relevant quantity.
$$
\begin{align*}
\mathrm{Var}[q(\mathbf{x}_{t-1}|\mathbf{x}_t)] & = \frac{\beta_t\tau_t}{\beta_t+\tau_t}\boldsymbol{I} = \frac{\tau_t}{1+\frac{\tau_t}{\beta_t}}\boldsymbol{I}
 \preceq \beta_t\boldsymbol{I}
\end{align*}
$$
This completes the proof.

**Discussion**. Should variances be learned instead of fixed? Ho et al. (2020) tried allowing the variances $\Sigma_\theta(\mathbf{x}_t,t)$ to be learned but found this led to worse reconstructions empirically. But Nichol and Dhariwal (2021) found that allowing it to be learned was necessary to achieve better log-likelihood estimates. They chose a non-isotropic but still diagonal variance parameterization, where each dimension independently interpolates between the upper and lower bounds previously described.
$$
\begin{align*}
\Sigma_\theta(\mathbf{x}_t,t) & = \mathrm{diag}\left(\exp\left(\mathbf{v}\log\beta_t + (1-\mathbf{v})\log \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\right)\right),\quad v_i\in[0,1]
\end{align*}
$$

For the sake of simplicity, this is not implemented here and we assume forward model variances are fixed in our codebase.

---

#### IV. Training

#### Simplified Loss

Instead of the actual ELBO, in the end we make two modifications.

1. Ignore $L_1 = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)]$.
2. Ignore weighting coefficients in the loss.

This leads to a simple loss that is optimized instead. For our hypothesized forward models,

$$
\begin{align*}
L_\mathrm{simple}(\mathbf{x}_0,t)^\text{(option 2)} & = \mathbb{E}_{q(\boldsymbol{\mathbf{x}_t}|\mathbf{x}_0)}\left\|\mathbf{x}_0 - f_\theta\left(\mathbf{x}_t, t \right)\right\|^2\\
& = \mathbb{E}_{q(\boldsymbol\epsilon)}\left\| \mathbf{x}_0 - f_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,t)\right\|^2\\
L_\mathrm{simple}(\mathbf{x}_0,t)^\text{(option 3)} & = \mathbb{E}_{q(\boldsymbol\epsilon)}\left\|\boldsymbol\epsilon -\epsilon_\theta\left(\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon, t \right)\right\|^2
\end{align*}
$$

The simple loss is equivalent to a re-weighting of the original ELBO. Ho et al. (2020) found that it produces better samples in practice compared to optimization of the true ELBO. Song et al. (2021) justified the use of this loss with the following result.

**Result**. Suppose the parameters for $f_\theta(\mathbf{x}_t,t)$ or $\epsilon_\theta(\mathbf{x}_t,t)$ are *not* shared across $t$. Then optimizing $L_\text{simple}$ will lead to the same solution as optimizing $L_t$. This is because optimizing each term independently is the same as optimizing the weighted sum.

So if our model is sufficiently powerful, in theory ignoring these weighting factors is okay.

Putting it together, we have the following training process. We will optimize this function over random samples of the groundtruth $\mathbf{x}_0$ and the timestep $t$.

```python
def compute_L_t(x_0, t):
    eps = sample N(0, I)
    monte_carlo_x_t = (
        sqrt_bar_alpha[t] * x_0 + sqrt_one_minus_bar_alpha[t] * eps
    )
    pred_eps = model(monte_carlo_x_t, t)  # gradient flows into model here
    loss = ((eps - pred_eps) ** 2).sum()
    return loss
```

---

#### Modeling Choices

What do we use for the model?

We need a mapping $f_\theta(\mathbf{x}_t,t) \mapsto \mathbb{R}^d$. In our simplest experiments we can use a feed-forward network. For images, though, a natural choice is a *U-Net*. The architecture of the U-Net is a series of ResNet blocks arranged into a set of downsamples following by upsamples. Ho et al. (2020) make two modifications to this basic idea:

1. To promote global sharing of features (rather than relying on local convolutions), they add self-attention layers.
2. To embed $t$, they take a corresponding sinusoidal embedding (Vaswani et al. 2017) and send it through a feed-forward network to modify the biases and scales of intermediate activations in each ResNet block.

We provide a simple implementation of a U-Net modified in this way [[here]](github.com/tonyduan/diffusion/src/models/blocks.p y). Below is the code for the ResNet block, which I found helpful for understanding.

```python

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x, t):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out

```

---

#### V. Accelerated Sampling

It's straightforward to sample the forward process.

```python
def sample():
    x = sample N(0, I)
    for t in T ... 1:
        noise = sample N(0, I) if t > 1 else 0
        pred_x_0 = model(x, t)
        x = (
            sqrt_bar_alpha[t-1] * beta[t] * pred_x_0 +
            (1 - bar_alpha[t-1]) * sqrt_alpha[t] * x
        ) / (1 - bar_alpha[t])
        x += sqrt_sigma * noise
    return x
```

But this is extremely slow when $T$ is large. How can we speed it up?

Let's revisit the assumed graphical model, call it $\mathcal{M}$.

$$
\begin{equation*}
\mathbf{x}_T \rightarrow \dots \rightarrow \mathbf{x}_{t} \rightarrow \mathbf{x}_{t-1} \rightarrow \dots \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_0 \tag{\(\mathcal{M}\)}
\end{equation*}
$$

The approach we'll take is sampling with *striding*. For the sake of concreteness, in the rest of this section we'll assume we want to sample every other latent variable (assume $T$ is even). That is, we want to start from $\mathbf{x}_T$, then sample $\mathbf{x}_{T-2}$, then $\mathbf{x}_{T-4}$, and so on until $\mathbf{x}_2$ and $\mathbf{x}_0$. Generalization to other, potentially non-uniform striding choices is straightforward and omitted for brevity.

One attempt would be to compute $q(\mathbf{x}_{t-2}|\mathbf{x}_t,\mathbf{x}_0)$ on $\mathcal{M}$. But the integral here is intractable to compute.

$$
\begin{align*}
q(\mathbf{x}_{t-2}|\mathbf{x}_t,\mathbf{x}_0) & = \int_{\mathbf{x}_{t-1}}q(\mathbf{x}_{t-2},\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})d\mathbf{x}_{t-1}\\
& = \int_{\mathbf{x}_{t-1}}q(\mathbf{x}_{t-2}|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0})d\mathbf{x}_{t-1}
\end{align*}
$$

Instead, we can hypothesize an alternative graphical model $\mathcal{M}^\ast$ which looks like the following.

$$
\begin{equation*}
\mathbf{x}_T \rightarrow \dots \rightarrow \mathbf{x}_{t} \rightarrow \mathbf{x}_{t-2} \rightarrow \dots \rightarrow \mathbf{x}_2 \rightarrow \mathbf{x}_0 \leftarrow \mathbf{x}_{1,3,5,\dots,T-1}\tag{\(\mathcal{M}^\ast\)}
\end{equation*}
$$

That is, the even-indexed variables form a Markov chain and the odd-indexed variables form a star graph connected to $\mathbf{x}_0$.

The key here is that we'll make distributional assumptions so that $\mathcal{M}^\ast$ has the *same marginal distributions* (conditioned on $\mathbf{x}_0$) as $\mathcal{M}$. Specifically, we'll assume the following.

**Define**. The accelerated backward process is the following.
$$
\begin{align*}
q(\mathbf{x}_{t}|\mathbf{x}_{t-2}) & = N\left(\sqrt\frac{\bar\alpha_t}{\bar\alpha_{t-2}}\mathbf{x}_{t-2},\left(1- \frac{\bar\alpha_t}{\bar\alpha_{t-2}}\right)\mathbf{I}\right)&& t \text{ is even}\\
q(\mathbf{x}_t|\mathbf{x}_0) & = N(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I}) && t \text{ is odd}
\end{align*}
$$

If we repeat the math previously derived for the backward model, we can verify that the marginal distributions of variables (conditioned on $\mathbf{x}_0$) match the original model. But the posterior distribution of the backward model will be different. Repeating the math will yield:

$$
\begin{align*}
q(\mathbf{x}_t|\mathbf{x}_0) & = N(\sqrt{\bar\alpha_t} \mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})&& \\
q(\mathbf{x}_{t-2}|\mathbf{x}_t,\mathbf{x}_0) & = N\left(\frac{\sqrt{\bar\alpha_{t-2}}}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_{t}}{\bar\alpha_{t-2}}\right)\mathbf{x}_0+ \frac{(1-\bar\alpha_{t-2})}{1-\bar\alpha_t}\sqrt{\frac{\bar\alpha_t}{\bar\alpha_{t-2}}}\mathbf{x}_t,\frac{1-\bar\alpha_{t-2}}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_{t}}{\bar\alpha_{t-2}}\right)\mathbf{I}\right)
\end{align*}
$$

**Define**. Our choice of hypothesized forward models will be more or less the same, as before, with modifications made intuitively.
$$
\begin{align*}
p_\theta(\mathbf{x}_{t-2}|\mathbf{x}_t)^{(\text{option }2)} & = N\left(\ \underbrace{\mathbb{E}\left[q(\mathbf{x}_{t-2}|\mathbf{x}_t,f_\theta(\mathbf{x}_t, t))\right]}_{\mu_\theta(\mathbf{x}_t, t)},\sigma_t^2 \mathbf{I}\right)\\
p_\theta(\mathbf{x}_{t-2}|\mathbf{x}_t)^{(\text{option }3)} & = N\left(\ \underbrace{\mathbb{E}\left[q\left(\mathbf{x}_{t-2}|\mathbf{x}_t,\frac{1}{\sqrt{\bar\alpha_t}}\left(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(\mathbf{x}_t, t)\right)\right)\right]}_{\mu_\theta(\mathbf{x}_t, t)}, \sigma_t^2 \mathbf{I}\right)
\end{align*}
$$

**Result**. The ELBO between this accelerated model and the original model will not be *exactly* the same; the weighting coefficients will be different due to the different posterior distribution $q(\mathbf{x}_{t-2}|\mathbf{x}_t,\mathbf{x}_0)$. But since we train with the simplified objective anyway we can ignore these weighting coefficients. We can therefore interpret the $\theta$ which we've already trained on $\mathcal{M}$ as corresponding to *this alterative graphical model* $\mathcal{M}^\ast$. And perform accelerated sampling by running the forward process on $\mathcal{M}^\ast$.

This approach was independently introduced by Song et al. (2021) and Nichol and Dhariwal (2021).

---

#### References

1. Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models. in *Advances in Neural Information Processing Systems* vol. 33 6840-6851 (2020).
2. Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. in *International Conference on Learning Representations* (2014).
3. Nichol, A. Q. & Dhariwal, P. Improved Denoising Diffusion Probabilistic Models. in *Proceedings of the 38th International Conference on Machine Learning* 8162-8171 (PMLR, 2021)
4. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. in *Proceedings of the 32nd International Conference on Machine Learning* 2256-2265 (PMLR, 2015).
5. Song, J., Meng, C. & Ermon, S. Denoising Diffusion Implicit Models. in *International Conference on Learning Representations* (2021).
6. Vaswani, A. *et al.* Attention is All you Need. in *Advances in Neural Information Processing Systems 30* (eds. Guyon, I. et al.) 5998-6008 (Curran Associates, Inc., 2017).
