import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        data=np.array(X)
        data.T
        xTx = data.T.dot(data)
        XtX = np.linalg.pinv(xTx)
        XtX_xT = XtX.dot(data.T)
        self.theta = XtX_xT.dot(y)
        return self.theta
        # *** END CODE HERE ***

    def fit_GD(self, X, y, alpha_value=0.01, iters=100):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        counter = 0
        if self.theta == None:
            self.theta = [0] * np.shape(X)[1]
        while (counter < iters):
            for j in range(len(self.theta)):
                predict = (self.predict(X)-y) * X[:,j]
                para_delta = -alpha_value * (sum(predict))
                self.theta[j]=self.theta[j] + para_delta
            counter += 1
        # *** END CODE HERE ***

    def fit_SGD(self, X, y, alpha_value=0.01, iters=100):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        counter=0
        if self.theta==None:
            self.theta = [0] * np.shape(X)[1]
        while (counter < iters):
            for i in range(len(X[:,0])):
                for j in range(len(self.theta)):
                    predict = (X[i].dot(self.theta) - y[i])
                    para_delta = alpha_value * predict * X[i][j]
                    self.theta[j] = self.theta[j] - para_delta
            counter += 1
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        output_array = []
        for i in range ( len ( X [:,1] ) ):
            x = X[i][1]
            output_array.append([x**l for l in range(k+1)])
        return np.array(output_array)
        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        Generates a cosine with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        output_array=[]
        for i in range ( len ( X[:,1] ) ):
            x=X[i][1]
            poly=[x**l for l in range(k+1)]
            y = 1.5 * np.cos(13 * x)
            poly.append(y)
            output_array.append(poly)
        return (np.array(output_array))
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        output_array = []
        for i in range ( len(X[:,0]) ):
            sum_predict=0
            for j in range(len(self.theta)):
                sum_predict += X[i][j] * self.theta[j]
            output_array.append(sum_predict)
        return output_array
        # *** END CODE HERE ***


    def predict_poly(self,X):
        output=[]
        #number of lines
        print(self.theta)

        for i in range (len(X)):
            sum_of_predictions=0
            for j in range(len(self.theta)):
                sum_of_predictions+=(X[i]**j)*self.theta[j]
            #print(sum_of_predictions)
            output.append(sum_of_predictions)
        return output

    def predict_cosine(self, X):
        output=[]
        #number of lines
        print(self.theta)

        for i in range (len(X)):
            sum_of_predictions=0

            for j in range(len(self.theta)):
                if j==len(self.theta)-1:
                    sum_of_predictions+=(np.cos(X[i]))*self.theta[j]
                else:
                    sum_of_predictions+=(X[i]**j)*self.theta[j]

            #print(sum_of_predictions)
            output.append(sum_of_predictions)

        return output

def run_exp(train_path, cosine=False, ks=[3, 5, 10, 20], filename='plot.pdf',fit_type='normal',output=True):

    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        if cosine==False:
            if fit_type == 'normal':
                model = LinearModel([0]*(k+1))
                training_data = model.create_poly(k,train_x)
                model.fit(training_data,train_y)
            if fit_type=='GD':
                model=LinearModel([0]*(k+1))
                training_data=model.create_poly(k,train_x)
                model.fit_GD(training_data,train_y)
            if fit_type=='SGD':
                model=LinearModel([0]*(k+1))
                training_data=model.create_poly(k,train_x)
                model.fit_SGD(training_data,train_y)
            plot_y = model.predict_poly(plot_x[:, 1])
        else:
            if fit_type=='normal':
                model=LinearModel([0]*(k+2))
                training_data=model.create_cosine(k,train_x)
                model.fit(training_data,train_y)
            if fit_type=='GD':
                model=LinearModel([0]*(k+2))
                training_data=model.create_cosine(k,train_x)
                model.fit_GD(training_data,train_y)
            if fit_type=='SGD':
                model=LinearModel([0]*(k+2))
                training_data=model.create_cosine(k,train_x)
                model.fit_SGD(training_data,train_y)
            plot_y = model.predict_cosine(plot_x[:, 1])
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2.5, 2.5)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    return(plot_x[:,1],plot_y)

def main(medium_path, small_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # A1 Q1 Part 1.2
    model = LinearModel()
    train_x,train_y = util.load_dataset(medium_path,add_intercept=True)
    run_exp(medium_path,cosine=False,ks=[3],filename='1.2_degree-3_polynomial_regression.png',fit_type='normal')

    # A1 Q1 Part 1.3
    normal_model = LinearModel()
    gd_model = LinearModel()
    sgd_model = LinearModel()
    train_x,train_y = util.load_dataset(medium_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    ndx, ndy = run_exp(medium_path,cosine=False,ks=[3],fit_type='normal',output=False)
    gdx, gdy = run_exp(medium_path,cosine=False,ks=[3],fit_type='GD',output=False)
    sdx, sdy = run_exp(medium_path,cosine=False,ks=[3],fit_type='SGD',output=False)
    plt.scatter(train_x[:,1], train_y)
    plt.plot(ndx,ndy,label='normal_fit')
    plt.plot(gdx,gdy,label='GD_fit')
    plt.plot(sdx,sdy,label='SGD_fit')
    plt.ylim(-2,2)
    plt.legend()
    plt.savefig('1.3_degree-3_polynomial_GD_and_SGD.png')

    # A1 Q1 Part 1.4
    run_exp(medium_path, cosine=False,filename='1.4_degree-3_normal_fit_polynomial.png',fit_type='normal')
    run_exp(medium_path, cosine=False,filename='1.4_degree-3_GD_fit_polynomial.png',fit_type='GD')
    run_exp(medium_path, cosine=False,filename='1.4_degree-3_SGD_fit_polynomial.png',fit_type='SGD')

    # A1 Q1 Part 1.5
    run_exp(medium_path, cosine=True,filename='1.5_other_feature_normal_fit_polynomial_cosine.png',fit_type='normal')
    run_exp(medium_path, cosine=True,filename='1.5_other_feature_GD_fit_polynomial_cosine.png',fit_type='GD')
    run_exp(medium_path, cosine=True,filename='1.5_other_feature_SGD_fit_polynomial_cosine.png',fit_type='SGD')

    # A1 Q1 Part 1.6
    run_exp(small_path, cosine=False,filename='1.6_overfitting_normal_fit_polynomial_cosine.png',fit_type='normal')
    run_exp(small_path, cosine=False,filename='1.6_overfitting_GD_fit_polynomial_cosine.png',fit_type='GD')
    run_exp(small_path, cosine=False,filename='1.6_overfitting_SGD_fit_polynomial_cosine.png',fit_type='SGD')
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
