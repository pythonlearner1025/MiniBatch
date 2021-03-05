import numpy as np
import cupy as cp
import functools
from timeit import default_timer as timer
from pipeline.pipeline import Pipe

class MiniBatch(Pipe):

    def __init__(self, batch=32, epoch=100, hidden_layer_num=5,
                 hidden_layer_size=25, output_layer_size=10, alpha=0.5, Lambda=None):
        self.batch = batch
        self.epoch = epoch
        self.alpha = alpha
        self.Lambda = Lambda
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_num = hidden_layer_num
        self.output_layer_size = output_layer_size
        self.batch_num = None
        self.batch_x = None
        self.batch_y = None
        self.Xtrain = None
        self.ytrain = None
        self.weights_list = None
        self.newvec = None
        self.final_weights = None

    def initiate(self, Xtrain, ytrain):
        self.init_weights(Xtrain, ytrain)
        placeholder, final_weights = self.gradient_descent()
        return final_weights

    def init_weights(self, Xtrain, ytrain):
        layers = 2 + self.hidden_layer_num
        if layers == 2:
            raise ValueError("Required Minimum 1 Hidden Layer")
        else:
            pass
        weights_dic = {}
        epi = cp.sqrt(6 / (self.hidden_layer_size + Xtrain.shape[1]))
        for i in range(1, layers):
            if i == 1:
                weights_dic['theta{}'.format(i)] \
                    = cp.c_[cp.ones((self.hidden_layer_size,1)), cp.random.rand(self.hidden_layer_size, Xtrain.shape[1]-1)] * (2 * epi) - epi
            if i == layers - 1:
                weights_dic['theta{}'.format(i)] \
                    = cp.random.rand(self.output_layer_size, self.hidden_layer_size) * (2 * epi) - epi
            elif 1 < i < (layers - 1):
                weights_dic['theta{}'.format(i)] \
                    = cp.random.rand(self.hidden_layer_size, self.hidden_layer_size) * (2 * epi) - epi
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        weights_list = [self.cparray(v) for k, v in weights_dic.items()]# make into type cp.float32
        self.weights_list = weights_list
        return self.Xtrain, self.ytrain, self.weights_list

    def make_batches(self):
        batch_num = int(self.Xtrain.shape[0] / self.batch)
        if batch_num * self.batch < (self.Xtrain.shape[0] - 50):
            raise ValueError("please print a value other than {}".format(self.batch))
        batch_X = np.zeros((batch_num, self.batch, self.Xtrain.shape[1]))
        batch_y = np.zeros((batch_num, self.batch, 1))
        for i in range(1, batch_num + 1):
            before = (i - 1) * self.batch
            after = i * self.batch
            batch_X[i - 1:, :, :] = self.Xtrain[before:after, :]
            batch_y[i - 1:, :, :] = self.ytrain[before:after, :]
        self.batch_num = batch_num
        self.batch_x = self.cparray(batch_X) # make into type cp.float32
        self.batch_y = self.cparray(batch_y) # make into type cp.float32
        return self.batch_num, self.batch_x, self.batch_y

    def gradient_descent(self):
        start = timer()
        self.make_batches()
        self.make_newvec()
        use_weights = self.weights_list
        print('generating {} optimized weights'.format(self.hidden_layer_num+1))
        for epoch in range(self.epoch):
            if (epoch + 1) % 10 == 0:
                print('updating grads at epoch {}'.format((epoch + 1)))
            for i in range(self.batch_num):
                got_grads = self.gradient_calc(use_weights, i)
                use_weights = [a - (self.alpha * b) for a, b in list(zip(use_weights, got_grads))]
        self.final_weights = use_weights
        stop = timer()
        print('elapsed time(s): ', stop - start)
        return self.final_weights, use_weights

    def gradient_calc(self, weights, batch_number):

        transposed_x = self.batch_x.reshape(self.batch_num, self.batch, 1, self.Xtrain.shape[1])[batch_number][cp.newaxis]
        all_layer = self.forward_prop(weights, transposed_x)
        last_layer = all_layer[-1].reshape(self.batch, 1, self.output_layer_size)[cp.newaxis]
        newvec_reshaped = self.newvec[batch_number].reshape(self.batch, 1, self.output_layer_size)[cp.newaxis]

        all_loss = self.calc_loss(all_layer, last_layer, newvec_reshaped, weights)
        all_loss = all_loss[::-1]

        if self.Lambda is None:
            gradient_sums = [cp.einsum('abdc', all_loss[i - 1], optimize='greedy') @ all_layer[i - 1] for i in range(1, self.hidden_layer_num + 2)]
            gradient_avg = [cp.einsum('hijk->jk', gradient, optimize='greedy') / self.batch for gradient in gradient_sums]
        elif self.Lambda is not None:
            gradient_sums = [cp.einsum('abdc', all_loss[i - 1], optimize='greedy') @ all_layer[i - 1] for i in
                             range(1, self.hidden_layer_num + 2)]
            gradient_avg = [cp.einsum('hijk->jk', gradient, optimize='greedy') / self.batch for gradient in
                            gradient_sums]
            gradient_avg = [grad + (self.Lambda/(self.batch*self.batch_num)) * weight for grad,weight in list(zip(gradient_avg,weights))]
        return gradient_avg

    def _score(self, Xtest=None, ytest=None):
        if Xtest is None and ytest is None:
            prediction = []
            for i in range(self.batch_num):
                transposed_x = self.batch_x.reshape(self.batch_num, self.batch, 1, self.Xtrain.shape[1])[i][cp.newaxis]
                prediction.append(self.forward_prop(self.final_weights, transposed_x)[-1])
            max = []
            for i in prediction:
                for z in range(self.batch):
                    max.append(cp.argmax(i[0][z], axis=1) + 1)
            prediction = cp.array(max).flatten()
            correct = self.batch_y.flatten()
            ans = sum(prediction == correct) / (self.batch_num * self.batch) * 100
        elif Xtest is not None and ytest is not None:
            batch_num = Xtest.shape[0]
            prediction = []
            for i in range(batch_num):
                transposed_x = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])[i][cp.newaxis][cp.newaxis]
                prediction.append(self.forward_prop(self.final_weights, transposed_x)[-1])
            max = [cp.argmax(i, axis=3)+1 for i in prediction]
            prediction = cp.array(max).flatten()
            correct = self.cparray(ytest.flatten())
            percentcorrect = sum(prediction == correct) / batch_num * 100
        return percentcorrect

    def _predict(self, Xtest):
        get_shape = len(Xtest.shape)
        if get_shape > 4:
            raise ValueError("This library only supports X,Y tensors up to dimension 3")
        elif get_shape <= 4:
            transposed_x = Xtest.reshape(1 , Xtest.shape[int(get_shape-1)])[cp.newaxis][cp.newaxis]
        prediction = self.cparray(self.forward_prop(self.final_weights, transposed_x)[-1])
        return cp.argmax(prediction[0][0], axis=1)

    def _forward_prop(func):
        @functools.wraps(func)
        def adding_layers(*args, **kwargs):
            setter, transposed_x, weights = func(*args, **kwargs)
            adding_layers.num_calls += 1
            nums = adding_layers.num_calls
            layers = adding_layers.layers
            if setter is True:
                if nums == 1:
                    layers.append(transposed_x)
                    layers.append(cp.maximum(0, (cp.einsum('ak,hijk->hija', weights[0], transposed_x, optimize='greedy'))))
                elif nums >= 2:
                    layers.append(cp.maximum(0, (cp.einsum('ak,hijk->hija', weights[nums - 1], layers[nums - 1], optimize='greedy'))))
            elif setter is False:
                adding_layers.num_calls = 0
                adding_layers.layers.clear()
            return layers
        adding_layers.layers = []
        adding_layers.num_calls = 0
        return adding_layers

    @_forward_prop
    def call_forward_prop(self, transposed_x, weights, setter=True):
        return setter, transposed_x, weights

    def forward_prop(self, weights, transposed_x):
        self.call_forward_prop(transposed_x, weights, False)
        for i in range(self.hidden_layer_num + 1):
            layers_all = self.call_forward_prop(transposed_x, weights, True)
        return layers_all

    def _calc_loss(func):
        @functools.wraps(func)
        def _calc(*args, **kwargs):
            def diffrelu(x):
                step = cp.maximum(0, x)
                return cp.where(step != 0, 1, step)
            setter, weights_list, last_layer, all_layer, newvec = func(*args, **kwargs)
            _calc.num_calls += 1
            nums = _calc.num_calls
            loss = _calc.loss
            if setter is True:
                if nums == 1:
                    loss.append(last_layer - newvec)
                if nums >= 2:
                    loss.append(cp.einsum('ak,hijk->hija', weights_list[-nums + 1].T, loss[nums - 2], optimize='greedy') * \
                                diffrelu(all_layer[-nums]))
            elif setter is False:
                _calc.num_calls = 0
                _calc.loss.clear()
            else:
                pass
            return loss
        _calc.num_calls = 0
        _calc.loss = []
        return _calc

    @_calc_loss
    def call_loss(self, all_layer, last_layer, newvec_reshaped, weights, setter=True):
        return setter, weights, last_layer, all_layer, newvec_reshaped

    def calc_loss(self, all_layer, last_layer, newvec_reshaped, weights):
        self.call_loss(all_layer,last_layer, newvec_reshaped, weights, False)
        for i in range(self.hidden_layer_num + 1):
            loss_all = self.call_loss(all_layer, last_layer, newvec_reshaped, weights, True)
        return loss_all

    def make_vec(self, newvec, batch_num):
        for i in range(1, 11):
            newvec[batch_num,:,i - 1][:, cp.newaxis] = cp.where(self.batch_y[batch_num,:,]==i, 1, 0)
        return newvec

    def make_newvec(self):
        newvec = self.cparray(cp.zeros((self.batch_num, self.batch, self.output_layer_size)))
        batch_num = -1
        while batch_num < (self.batch_num - 1):
            batch_num += 1
            newvec = self.make_vec(newvec, batch_num)
        self.newvec = newvec
        return self.newvec

    def cparray(self, *args, **kwargs):
        kwargs.setdefault("dtype", cp.float32)
        return cp.array(*args, **kwargs)

    def nparray(self, *args, **kwargs):
        kwargs.setdefault("dtype", np.float32)
        return np.array(*args, **kwargs)





















