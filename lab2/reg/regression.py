import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    # Generate a grid of points (and formatting) for plotting the contour of the prior distribution

    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0, a1)
    grid_points = np.c_[A0.ravel(), A1.ravel()]

    mu_prior = np.array([0, 0])  # Mean of the prior distribution
    cov_prior = np.array([[beta, 0], [0, beta]])  # Covariance of the prior distribution

    density = util.density_Gaussian(mu_prior, cov_prior, grid_points).reshape(A0.shape)

    plt.figure()
    plt.contour(A0, A1, density, colors='blue')
    plt.plot(-0.9, 0.9, 'rx', markersize=10, markeredgewidth=2, label='True a')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title('Prior Distribution p(a)')
    plt.legend()
    plt.savefig('prior.pdf')
    plt.close()

    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    ns = x.shape[0] # number of training samples used to compute posterior

    X_aug = np.hstack((np.ones((ns, 1)), x)) 

    #posterior distribution p(a|x,z) is also Gaussian, with mean mu and covariance Cov
    I = np.eye(2)
    Cov = np.linalg.inv((1/beta) * I + (1/sigma2) * np.dot(X_aug.T, X_aug))


    mu_col = (1/sigma2) * np.dot(Cov, np.dot(X_aug.T, z))
    mu = mu_col.flatten() # convert to 1D for util function

    #gridspoints for plotting the contour of the posterior distribution
    a0 = np.linspace(-1, 1, 100)
    a1 = np.linspace(-1, 1, 100)
    A0, A1 = np.meshgrid(a0, a1)
    grid_points = np.c_[A0.ravel(), A1.ravel()]
    
    density = util.density_Gaussian(mu, Cov, grid_points).reshape(A0.shape)

    # plots
    plt.figure()
    plt.contour(A0, A1, density, colors='blue')
    plt.plot(-0.9, 0.9, 'rx', markersize=10, markeredgewidth=2, label='True a')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title(f'Posterior Distribution p(a|z) for N={ns}')
    plt.legend()
    plt.savefig(f'posterior{ns}.pdf')
    plt.close()


    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    ns = x_train.shape[0]
    
    means = []
    stds = []

    for xi in x:
        x_vec = np.array([1, xi]) # Augment the test point
        
        # Predictive mean
        pred_mean = np.dot(mu, x_vec)
        
        # Predictive variance
        pred_var = np.dot(x_vec.T, np.dot(Cov, x_vec)) + sigma2
        
        means.append(pred_mean)
        stds.append(np.sqrt(pred_var))
        
    means = np.array(means)
    stds = np.array(stds)
    
    plt.figure()
    
    #  Plot the mean prediction line AND the error bars 
    plt.errorbar(x, means, yerr=stds, fmt='r-', ecolor='red', elinewidth=1, capsize=3, label='Prediction \u00b1 1 std')
    
    # 3. Plot the training data 
    plt.scatter(x_train.flatten(), z_train.flatten(), c='blue', marker='o', s=50, label='Training Data', zorder=10)
    
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('Input x')
    plt.ylabel('Target z')
    plt.title(f'Predictive Distribution for N={ns}')
    plt.legend(loc='lower right')
    plt.savefig(f'predict{ns}.pdf')
    plt.close()

    
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
