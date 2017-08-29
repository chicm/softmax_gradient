# Calculate gradient of cross entropy loss

Let's say we have a nueron network with softmax classifier at the last layer, using cross entropy loss function.

> Softmax function is defined as following:

>  $$ p_j = \dfrac{e^j}{\displaystyle\sum_{i} e^i} $$ 

Cross entropy loss function is defined as following:

> $$ L = -\displaystyle\sum_{i} y_ilog(p_i) $$

For example, suppose we have 3 classes, one class score for a specific input X before applying softmax is:

> $$ o = \boxed{2}\boxed{3}\boxed{4} $$
o is the output of last layer before applying softmax activation function.

The label y for X is:
> $$ y = \boxed{0}\boxed{1}\boxed{0} $$
which means the second class (class 1) is the correct class.

After applying softmax function to the class score, we get the predicted probablities p as following:
> $$ p = \boxed{0.090}\boxed{0.245}\boxed{0.665} $$

Assume the index of correct class is k (here in this exmple, k=1), for incorrect class, the $y_i$ is always 0, therefore, the loss function can be simplified as:
> $$ L = -y_klog(p_k) = -log(p_k)$$
For this exmple:
> $$ L = -log(0.245) = 1.406 $$

Now let's calculate the gradient of loss function L with respect to class score $o$.

It would be easier to understand if we calculate the derivative of L with respect to correct class $o_k$ and incorrect class $o_j$ seperately.

Let's firstly calculate the derivative of L w.r.t $o_k$ :
> $$ \dfrac{\partial L}{\partial o_k} = \dfrac{\partial}{o_k}(-log(p_k)) = -\dfrac{\partial}{o_k}(log(p_k)) = - \dfrac{1}{p_k}\dfrac{\partial}{o_k}(p_k)$$
> $$ =  - \dfrac{1}{p_k}\dfrac{\partial}{o_k}\Bigg(\dfrac{e^{o_k}}{\displaystyle\sum_{i} e^{o_i}}\Bigg) = - \dfrac{1}{p_k} \dfrac{e^{o_k}{\displaystyle\sum_{i} e^{o_i}}-\Big(\dfrac{\partial}{o_k}\displaystyle\sum_{i} e^{o_i}\Big)e^{o_k}}{\Big({\displaystyle\sum_{i} e^{o_i}}\Big)^2}$$

>$$ = - \dfrac{1}{p_k} \Bigg( \dfrac{e^{o_k}}{\displaystyle\sum_{i} e^{o_i}} - \dfrac{e^{o_k}\dfrac{\partial}{o_k}\displaystyle\sum_{i} e^{o_i}}{\Big({\displaystyle\sum_{i} e^{o_i}}\Big)^2}\Bigg) $$

>$$ = - \dfrac{1}{p_k} \Bigg( p_k - \dfrac{e^{o_k}\dfrac{\partial}{o_k}\displaystyle\sum_{i} e^{o_i}}{\Big({\displaystyle\sum_{i} e^{o_i}}\Big)^2}\Bigg) $$

Now let's calculate this part: $\dfrac{\partial}{o_k}\displaystyle\sum_{i} e^{o_i}$, $\forall i\ne k$, $e^{o_i}$ just constants, so $\dfrac{\partial}{o_k}\displaystyle\sum_{i} e^{o_i} = e^{o_k}$
then we have:
>$$ \dfrac{\partial L}{\partial o_k} =  - \dfrac{1}{p_k} \Bigg( p_k - \dfrac{(e^{o_k})^2}{\Big({\displaystyle\sum_{i} e^{o_i}}\Big)^2}\Bigg) = - \dfrac{1}{p_k} \Bigg( p_k - (p_k)^2\Bigg) = p_k - 1 $$ 

For incorrect class $j$, $e^{o_k}$ is a constant,
> $$ \dfrac{\partial L}{\partial o_j} = \dfrac{\partial}{o_j}(-log(p_k)) = -\dfrac{\partial}{o_j}(log(p_k)) = - \dfrac{1}{p_k}\dfrac{\partial}{o_j}(p_k)$$
> $$ =  - \dfrac{1}{p_k}\dfrac{\partial}{o_j}\Bigg(\dfrac{e^{o_k}}{\displaystyle\sum_{i} e^{o_i}}\Bigg) = - \dfrac{e^{o_k}}{p_k}\dfrac{\partial}{o_j}\Bigg(\dfrac{1}{\displaystyle\sum_{i} e^{o_i}}\Bigg)$$
> $$ = - \dfrac{e^{o_k}}{p_k} (-1) \dfrac{1}{\Big(\displaystyle\sum_{i} e^{o_i}\Big)^2} \dfrac{\partial}{o_j}\displaystyle\sum_{i} e^{o_i} $$
> $$ =  \dfrac{e^{o_k}}{p_k} \dfrac{1}{\Big(\displaystyle\sum_{i} e^{o_i}\Big)^2}  e^{o_j} =  \dfrac{1}{p_k} \dfrac{e^{o_k}}{\displaystyle\sum_{i} e^{o_i}} \dfrac{e^{o_j}}{\displaystyle\sum_{i} e^{o_i}} $$
>$$ = \dfrac{1}{p_k} p_k p_j = p_j $$

Finally, for the exmample, we found the gradient of L w.r.t to p is:

$$ gradients = \boxed{0.090}\boxed{-0.775}\boxed{0.665} $$
