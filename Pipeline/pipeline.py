from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io import loadmat

class Pipe:
    def __init__(self, steps):
        self.steps = steps
        self.estimators = steps
        self.params = None
        self.Xtrain = None
        self.ytrain = None
        self.Ctrain = None
        self.Xtest = None
        self.ytest = None

    def use_estimator(self, input_estimators, Xtrain, ytrain):
        from pipeline.minidescent.fast import MiniBatch
        valid_class = [StandardScaler(), MiniBatch()]
        num_attributes = len(input_estimators)
        for i in range(num_attributes):
            setattr(self, 'attributenumber' + str(i), input_estimators[i])
        if num_attributes <= 2:
            scaler = StandardScaler()
            model = self.attributenumber1
            params = model.initiate(Xtrain, ytrain)
        else:
            raise ValueError("Sorry, only support 2 for now")
        return params

    def fit(self, Xtrain=None, ytrain=None, cv=False):
        if cv is False:
            if Xtrain is None and ytrain is None:
                Xtrain = self.Xtrain
                ytrain = self.ytrain
            elif (Xtrain is None and ytrain is not None) or (Xtrain is not None and ytrain is None):
                raise ValueError('Either use load() to attempt to create new instances of data, and use fit() without args,')
        elif cv is True:
            if Xtrain is None and ytrain is None:
                training = int(0.8 * self.Xtrain.shape[0])
                ok = int((self.Xtrain.shape[0] - training) / 2)
                Xtrain = self.Xtrain[:training]
                ytrain = self.ytrain[:training]
                self.Ctrain = self.Xtrain[training:training + ok]
                self.Xtest = self.Xtrain[training + ok:]
                self.ytest = self.ytrain[training + ok:]
            elif (Xtrain is None and ytrain is not None) or (Xtrain is not None and ytrain is None):
                raise ValueError(
                    'Either use load() to attempt to create new instances of data, and use fit() without args,')
        self.params = self.use_estimator(self.estimators, Xtrain, ytrain)
        return self.params, self.Ctrain, self.Xtest, self.ytest

    def coefs(self):
        if self.params == 0:
            raise ValueError("Use the fit() method to generate optimized weights first")
        else:
            pass
        return print('fitted weights: {}'.format(self.params))

    def score(self):
        if self.params == 0:
            raise ValueError("Use the fit() method to generate optimized weights first")
        else:
            estimators = self.steps
            num_attributes = len(self.steps)
            for i in range(num_attributes):
                setattr(self, 'attributenumber' + str(i), estimators[i])
            if num_attributes <= 2:
                model = self.attributenumber1
        if np.any(self.Xtest) is None:
            score = model._score(self.Xtrain, self.ytrain)
        elif np.any(self.Xtest) is not None:
            score = model._score(self.Xtest, self.ytest)
        return print('{} percent correct'.format(score))

    def predict(self, Xtest):
        estimators = self.steps
        num_attributes = len(self.steps)
        for i in range(num_attributes):
            setattr(self, 'attributenumber' + str(i), estimators[i])
        if num_attributes <= 2:
            model = self.attributenumber1
        prediction = model._predict(Xtest)
        return prediction

    def load(self, data, cv=None):
        try:
            loadmat(data)
        except FileNotFoundError:
            print("FileNotFoundError: Reference an existing File")
        except TypeError:
            print("TypeError: Enter a file of .mat format")
        else:
            local_file = loadmat(data)
        try:
            local_file['X']
            local_file['y']
        except KeyError:
            print("X,y not found in file")
        else:
            X = local_file['X']
            y = local_file['y']
            Xb = np.c_[np.ones((X.shape[0],1)), X]
        # shuffle
        placeholder = np.hstack((Xb, y))
        shuffled_all = placeholder[np.random.randint(Xb.shape[0], size=Xb.shape[0]),:]
        self.Xtrain = shuffled_all[:,:-1]
        self.ytrain = shuffled_all[:,-1:]
        return self.Xtrain, self.ytrain



