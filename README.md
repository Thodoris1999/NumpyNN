# NumpyNN
A small linear neural network interface for educational purposes.

I wanted to get a better intuition on backpropagation and how machine learning libraries work so I made this based on [this video](https://www.youtube.com/watch?v=tIeHLnjs5U8).

I chose a verbose naming scheme. The naming convention for arithmetic variables is as follows:
1. x is the input, y is the label, z[i] is the i-th neuron value without the activation, a[i] is the value on the i-th activation, b[i] is the i-th bias vector, w[i] is the i-th weight matrix
2. a[0] is the input x for indexing convenience
3. derivatives are prefixed by "d_"
4. "d_x_y" means "derivative of x with respect to y" (dx/dy). The "_x" part can be skipped, that means "derivative of the loss function with respect to x"
5. Arithmetic variables can have subscripts that are just placed next to them, that refer to which layer they refer to. "L" means the last layer

For example "d_al_zk" means "the derivatives of the activations in the l-th layer with respect to the neuron values without the activations in the k-th layer"
