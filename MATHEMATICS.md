# 🪐 StratOS-RL: Mathematical Formalization

StratOS-RL is modeled as a **Non-Stationary Stochastic Game** with lagged state transitions. This document formalizes the transition dynamics and reward surfaces.

## 1. Resource Allocation & Lags

Let $A_t = [m_t, p_t, i_t, h_t]$ be the action vector at time $t$ representing marketing, product, infra, and hiring spend.

### Human Capital Ramp-up
Hiring is subject to a 3-step discrete convolution:
$$T_{t+3} = T_t + \sum H_{t}$$
Where $T$ is productive team size and $H$ is the hiring queue. This creates a **Liquidity Gap** where expenses precede utility.

### Brand Momentum
Marketing spend $m_t$ generates immediate leads but also contributes to a decaying latent variable $B$ (Brand):
$$B_{t+1} = \gamma B_t + \lambda \ln(1 + m_t)$$
Where $\gamma \in [0, 1]$ is the persistence factor and $\lambda$ is the resonance coefficient.

## 2. Customer Acquisition Dynamics

The conversion rate $\eta$ is a multi-objective function:
$$\eta_t = f(P_t, Q_t, S_t, B_t)$$
- **Price Elasticity ($P$)**: Churn increases non-linearly as price exceeds market mean.
- **Product Quality ($Q$)**: Cumulative investment in R&D improves retention via a log-growth model.
- **Saturation ($S$)**: Acquisition cost (CAC) scales as market share $M$ approaches the Total Addressable Market (TAM).

## 3. The Insolvency Trap

Bankruptcy is not instant. It is defined by the **Insolvency Integral**:
$$I = \int_{t_0}^{t_{now}} \mathbb{1}(Cash_t < \text{Liability}_t) dt$$
If $I > 10$, the episode terminates. This forces the agent to manage "Lender Trust," where high burn rates lead to non-linear interest spikes:
$$\text{Interest}_t = \text{Debt}_t \times (0.025 + \alpha \cdot \text{BurnRatio}^2)$$

## 4. Reward Objective

The agent's objective is to maximize the **Discounted Strategy Score**:
$$J = \max \sum_{t=0}^{T} \beta^t R(MRR_t, \text{Cash}_t)$$
Where $\beta$ is the discount factor. The benchmark evaluates agents on their ability to survive the high-capital-intensity Year 1 to reach the high-LTV Year 5.
