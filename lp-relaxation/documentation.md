---
title: "LP Relaxation for Training Binary Neural Networks"
geometry: margin=2cm
author: Li-Ping Liu
output: pdf_document
---

# Introduction

In this work, we study methods for training large-scale Binary Neural Networks (BNNs). BNNs have the potential to significantly reduce the energy consumption of traditional neural networks that run calculations on floating-point numbers.

We focus on Pure Binary Neural Networks (PBNN), which run ALL calculations with binary
operations and do *not* use any integers or floating-point numbers. This is different 
from many BNN studies, where a small fraction of operations still run on floating-point 
or integer arithmetic. 

The focus on PBNN is motivated by hardware design. If a large-scale neural network (e.g. an LLM) can be implemented using pure binary operations, then the hardware design would be
greatly simplified. The calculation can then easily be run on an FPGA or an In-Memory Computing (IMC) device. 

At the same time, training a PBNN is notoriously hard. A PBNN is fundamentally a function of a large number of binary variables, and training a PBNN is essentially optimizing this function with respect to binary model parameters. It is well known that finding the global optimum is NP-hard and hopeless in general. However, the theory of binary optimization suggests that the optimization problem is not as hard when there is a large number of optimization variables, which gives the problem more freedom. Meanwhile, research on deep learning shows that a learning model still works well even when its parameters do not form the global optimum of the loss function. 

This work aims to develop a new optimization method for PBNN by combining stochastic training with a Linear-Programming (LP) relaxation of the binary optimization problem.

# Related Work 

[TODO: add a discussion of related work.]



# Method 

## Setting up the learning problem 

We assume an input instance is $\mathbf{x} \in \{0, 1\}^d$, and the class label is $y \in \{1, \ldots, C\}$. We consider a feedforward neural network parameterized by: 

* $W^{1}, W^{2}, W^{3}, ..., W^{T}$, with each $W^{t} \in \{0, 1\}^{d_{t} \times {d_{t-1}}}$; and  
* $b^{1}, b^{2}, b^{3}, ..., b^{T}$, with each $b^{t} \in \{0, 1\}^{d_{t}}$. 

Let $\odot$, $+$, and $\oplus$ respectively represent the AND, OR, and XOR operations. 
Let $\bigvee(\cdot)$ denote the or-reduce function, which takes OR over 
a list of variables.  
$$
\bigvee(z_1, \dots, z_n) = z_1 + z_2 + ... + z_n .
$$
 
Let $\mathbf{h}^0 = \mathbf{x}$, and each layer is computed by: 
$$
h^{t}_i = b^t_i ~\oplus~ \bigvee (W^{t}_{i, j} \odot h^{t-1}_{j}: j = 1, \ldots, d_{t-1})
$$

In the last layer, the prediction is $\mathbf{s} = \mathbf{h}^T$.
We obtain the loss by comparing $\mathbf{s}$ against $y$; here the computation is no longer limited to binary operations. We assume that the loss on this instance is: 
$$
L = \sum_{j=1}^{d_T} u_j s_j
$$
Here $u_j$ is a float constant decided by $y$. [TODO: need more work to properly define u_j from y]. 

From the perspective of computation, we could run the data loading with a CPU and then send the data to a special device running PBNN to get the prediction 
$\mathbf{s}$. Then the CPU can run further calculations to get the final prediction $\hat{y}$.  

## Minimize the objective with the LP relaxation 

Now we want to minimize the loss with respect to binary model parameters ($W$ matrices and $b$ vectors). LP relaxation is a well-known method for optimizing objectives defined on binary variables. To relax the problem into a continuous problem, we also include $\mathbf{h}^t$ for all $t$ as variables.

First, we need to express the relationship $\mathbf{h}^t$, $\mathbf{W}^t$, and $\mathbf{h}^{t-1}$ as a linear constraint. Let $g^t_i$ be the value after OR-reduce but before the XOR operation with $b^t_i$,
$$
g^{t}_i = \bigvee_{j=1}^{d_{t-1}} \left(W^{t}_{i,j} \odot h^{t-1}_{j}\right), \qquad h^{t}_i = b^{t}_i \oplus g^{t}_i.
$$

Relax all variables to the unit interval, with $\mathbf{h}^0 = \mathbf{x}$ fixed:
$$
W^{t}_{i,j} \in [0, 1], \qquad b^{t}_i \in [0, 1], \qquad g^{t}_i \in [0, 1], \qquad h^{t}_i \in [0, 1]
$$

For $t = 1, \ldots, T$, $i = 1, \ldots, d_t$, $j = 1, \ldots, d_{t-1}$, introduce $z^t_{i,j} = W^t_{i,j} \otimes h^{t-1}_j$. We linearize the AND operation with the (two-sided) McCormick envelope:
$$
z^{t}_{i,j} \le W^{t}_{i,j}, \qquad z^{t}_{i,j} \le h^{t-1}_{j}, \qquad z^{t}_{i,j} \ge W^{t}_{i,j} + h^{t-1}_{j} - 1, \qquad z^{t}_{i,j} \ge 0
$$

The OR-reduce operation can be relaxed to:
$$
g^{t}_i \ge z^{t}_{i,j} \ \ \forall j, \qquad g^{t}_i \le \sum_{j=1}^{d_{t-1}} z^{t}_{i,j}
$$

The linearization of $h^{t}_i = b^{t}_i \oplus g^{t}_i$ is: 
$$
h^{t}_i \le g^{t}_i + b^{t}_i, \qquad h^{t}_i \le 2 - g^{t}_i - b^{t}_i, \qquad h^{t}_i \ge g^{t}_i - b^{t}_i, \qquad h^{t}_i \ge b^{t}_i - g^{t}_i
$$

The resulting LP is:
$$
\min_{W, b, g, h, z} \quad L = \sum_{j=1}^{d_T} u_j h^{T}_j
$$
$$
\text{s.t.} \quad z^{t}_{i,j} \le W^{t}_{i,j}, \quad z^{t}_{i,j} \le h^{t-1}_{j}, \quad z^{t}_{i,j} \ge W^{t}_{i,j} + h^{t-1}_{j} - 1, \quad z^{t}_{i,j} \ge 0 \quad \forall t, i, j
$$
$$
g^{t}_i \ge z^{t}_{i,j} \ \ \forall t, i, j, \qquad g^{t}_i \le \sum_{j=1}^{d_{t-1}} z^{t}_{i,j} \quad \forall t, i
$$
$$
h^{t}_i \le g^{t}_i + b^{t}_i, \quad h^{t}_i \le 2 - g^{t}_i - b^{t}_i, \quad h^{t}_i \ge g^{t}_i - b^{t}_i, \quad h^{t}_i \ge b^{t}_i - g^{t}_i \quad \forall t, i
$$
$$
\mathbf{h}^0 = \mathbf{x}, \qquad W^{t}_{i,j} \in [0, 1], \qquad b^{t}_i \in [0, 1], \qquad g^{t}_i \in [0, 1]
$$

## Stochastic training

Let $\alpha = (W^1, \ldots, W^T, b^1, \ldots, b^T)$ denote the shared parameters, with $|\alpha| = T(d^2+d)$ under the $d_t = d$ simplification. For a training set of $N$ samples $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}$, let
$$
\beta^{(k)} = \left\{ z^{t,(k)}_{i,j},\ g^{t,(k)}_i,\ h^{t,(k)}_i \right\}_{t,i,j}
$$
denote the per-sample auxiliary variables introduced by the linearization above, with $h^{0,(k)} = \mathbf{x}^{(k)}$ fixed; each $\beta^{(k)}$ has $T(d^2+2d)$ entries. The full LP is
$$
\min_{\alpha, \beta^{(1)}, \ldots, \beta^{(N)}} \quad \sum_{k=1}^{N} \sum_{j=1}^{d_T} u_j\, h^{T,(k)}_j
\qquad \text{s.t.} \quad \text{constraints}\big(\alpha, \beta^{(k)}; \mathbf{x}^{(k)}\big) \quad \forall k = 1, \ldots, N,
$$
where "constraints" collects the AND, OR-reduce, and XOR linearizations from above, applied independently to each sample. This has $|\alpha| + N\cdot|\beta^{(1)}|$ variables in total.

**Exact decomposition.** The constraint blocks for different samples share only $\alpha$ and never interact directly, so the minimization over $\beta^{(1)}, \ldots, \beta^{(N)}$ separates exactly:

$$
\begin{aligned}
& \min_{\alpha,\ \beta^{(1:N)}} \sum_{k=1}^{N} \sum_j u_j h^{T,(k)}_j \quad \text{s.t.} \quad \text{constraints}\big(\alpha, \beta^{(k)}; \mathbf{x}^{(k)}\big) \ \ \forall k = 1, \ldots, N  \\
& = \min_{\alpha} \sum_{k=1}^N \underbrace{\min_{\beta^{(k)}} \Big\{ \textstyle\sum_j u_j h^{T,(k)}_j \ :\ \text{constraints}(\alpha, \beta^{(k)}; \mathbf{x}^{(k)}) \Big\}}_{=: V(\alpha;\, \mathbf{x}^{(k)})} \\
& = \min_{\alpha \in [0,1]^{|\alpha|}} \sum_{k=1}^N V(\alpha; \mathbf{x}^{(k)}).
\end{aligned}
$$
The only constraint on $\alpha$ is that it lies in the box $[0,1]^{|\alpha|}$. Therefore, we only need a subgradient of $\sum_{k=1}^N V(\alpha; \mathbf{x}^{(k)})$ to get the update direction for the parameters in $\alpha$. If an update step would move a parameter outside $[0,1]$, the parameter is instead projected back onto the boundary.  

The subgradient can be estimated using $V(\alpha; \mathbf{x}^{(k)})$ evaluated on a subset (mini-batch) of training examples. For each sample, we first compute the optimal $\beta^{(k)}$ (i.e. solve for $V(\alpha; \mathbf{x}^{(k)})$), and then take the subgradient of the result with respect to $\alpha$.


