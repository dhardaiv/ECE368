import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the 2D features of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_1,mu_2,cov,cov_1,cov_2
    in which mu_1, mu_2 are mean vectors (as 1D arrays)
             cov, cov_1, cov_2 are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    # 1. Filter the data into two classes
    male_indices = (y == 1)
    female_indices = (y == 2)
    
    x_male = x[male_indices]
    x_female = x[female_indices]
    
    N_male = x_male.shape[0]
    N_female = x_female.shape[0]
    N = x.shape[0]
    
    # Compute Mean Vectors (mu_1 and mu_2)
    mu_male = np.mean(x_male, axis=0)
    mu_female = np.mean(x_female, axis=0)
    
    # Compute Covariance Matrices
    cov_male = np.cov(x_male, rowvar=False, ddof=0)
    cov_female = np.cov(x_female, rowvar=False, ddof=0)
    
    # LDA Shared Covariance is the weighted average of the two
    cov = (N_male * cov_male + N_female * cov_female) / N
    
  
    # Visualizing LDA and QDA
  
    #  2D grid for plotting 
    x_range = np.linspace(-4, 6, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Flatten the grid into a list of (x, y) 
    grid_points = np.c_[X.ravel(), Y.ravel()]
    
    # ----- Plot LDA -----
    density_male_lda = util.density_Gaussian(mu_male, cov, grid_points).reshape(X.shape)
    density_female_lda = util.density_Gaussian(mu_female, cov, grid_points).reshape(X.shape)
    
    plt.figure()
    plt.scatter(x_male[:, 0], x_male[:, 1], c='blue', label='Class 1')
    plt.scatter(x_female[:, 0], x_female[:, 1], c='red', label='Class 2')
    
    # Plot contours and decision boundary 
    plt.contour(X, Y, density_male_lda, colors='blue', alpha=0.3)
    plt.contour(X, Y, density_female_lda, colors='red', alpha=0.3)
    plt.contour(X, Y, density_male_lda - density_female_lda, levels=[0], colors='black', linewidths=2)
    
    plt.title('Linear Discriminant Analysis (LDA)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('lda.pdf')
    plt.close()
    
    # ----- Plot QDA -----
    density_male_qda = util.density_Gaussian(mu_male, cov_male, grid_points).reshape(X.shape)
    density_female_qda = util.density_Gaussian(mu_female, cov_female, grid_points).reshape(X.shape)
    
    plt.figure()
    plt.scatter(x_male[:, 0], x_male[:, 1], c='blue', label='Class 1')
    plt.scatter(x_female[:, 0], x_female[:, 1], c='red', label='Class 2')
    
    # Plot contours and decision boundary (where densities are equal)
    plt.contour(X, Y, density_male_qda, colors='blue', alpha=0.3)
    plt.contour(X, Y, density_female_qda, colors='red', alpha=0.3)
    plt.contour(X, Y, density_male_qda - density_female_qda, levels=[0], colors='black', linewidths=2)
    
    plt.title('Quadratic Discriminant Analysis (QDA)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('qda.pdf')
    plt.close()

    return (mu_male, mu_female, cov, cov_male, cov_female)


def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_1,mu_2,cov,cov_1,mu_2: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the 2D features of the N samples 
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    
    N_test = x.shape[0]
    
 
    # LDA Misclassification Rate
   
    prob_male_lda = util.density_Gaussian(mu_male, cov, x)
    prob_female_lda = util.density_Gaussian(mu_female, cov, x)
    
    # assign predictions
    pred_lda = np.zeros(N_test)
    pred_lda[prob_male_lda >= prob_female_lda] = 1
    pred_lda[prob_female_lda > prob_male_lda] = 2
    
    # Count how many are wrong and divide by total
    errors_lda = np.sum(pred_lda != y)
    mis_lda = errors_lda / N_test
    


    #  QDA Misclassification Rate
    prob_male_qda = util.density_Gaussian(mu_male, cov_male, x)
    prob_female_qda = util.density_Gaussian(mu_female, cov_female, x)
    
    #  assign predictions
    pred_qda = np.zeros(N_test)
    pred_qda[prob_male_qda >= prob_female_qda] = 1
    pred_qda[prob_female_qda > prob_male_qda] = 2
    
    # Count how many are wrong 
    errors_qda = np.sum(pred_qda != y)
    mis_qda = errors_qda / N_test
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainData.txt')
    x_test, y_test = util.get_data_in_file('testData.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    
    print(f"LDA Misclassification Rate: {mis_LDA * 100:.2f}%")
    print(f"QDA Misclassification Rate: {mis_QDA * 100:.2f}%")

    
    
    

    
