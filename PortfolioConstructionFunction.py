#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:12:22 2017

@author: jagger.guo
"""

import pandas as pd
import numpy as np
import random
import string
from sklearn.linear_model import LinearRegression

# 1. Define the basic mean-variance model to construct portfolios

class Mean_Variance_Model(object):
    
    def __init__(self, data, n, rf):
        
        self.data = data # Initialize the input data
        self.n = n # Initialize the number of assets (features)
        self.rf = rf # Initialize the risk-free rate
    
    def r_mean_vec(self): # Create the expected return vector
        
        expected_return_vec = np.transpose(np.mat(self.data.mean()))
        
        return expected_return_vec
    
    def ex_r_mat(self): # Create the excess return matrix
        
        excess_return_matrix = np.mat(self.data - self.data.mean())
        
        return excess_return_matrix
    
    def get_cov_mat(self): # Create the sample covariance matrix
        
        ex_r_mat = self.ex_r_mat()
        cov_matrix = np.mat((1 / (len(ex_r_mat) - 1)) * np.dot(ex_r_mat.T, ex_r_mat))
        
        return cov_matrix
    
    def num_abcd(self): # Calculate the numbers of a, b, c, d
        
        ones_vec = np.ones((self.n, 1))
        cov_matrix = self.get_cov_mat()
        expected_return_vec = self.r_mean_vec()
        
        a = ones_vec.T * (cov_matrix.I) * ones_vec
        b = ones_vec.T * (cov_matrix.I) * expected_return_vec
        c = expected_return_vec.T * (cov_matrix.I) * expected_return_vec
        d = (a*c - b**2)
        
        return a, b, c, d
    
    def tp_x_vec(self): # Calculate the X vector, the expected return and standard deviation of the tangency portfolio 
       
        a, b, _, _ = self.num_abcd()
        risk_aversion = b - self.rf * a # Set the risk-free rate as 0 
        cov_mat = self.get_cov_mat()
        r_mean = self.r_mean_vec()
    
        x_vec = np.multiply(1/risk_aversion, np.dot(cov_mat.I, r_mean - 
                                                    (np.zeros((self.n,1)))))
        expected_return_tp = np.dot(r_mean.T, x_vec)
        sd_tp = np.sqrt(np.dot(np.dot(x_vec.T, cov_mat), x_vec))
    
        return x_vec, expected_return_tp, sd_tp
    
    def gmvp_x_vec(self): # Calculate the X vector, the expected return and standard deviation at the GMVP point
        
        a, b, _, _ = self.num_abcd()
        x_vec = np.dot(np.multiply(1/a, self.get_cov_mat().I), np.ones((self.n, 1)))
        expected_return_GMVP = np.array((b/a))
        sd_GMVP = np.array(np.sqrt(1/a))
    
        return x_vec, expected_return_GMVP, sd_GMVP
    
    
# 2. Define the factor model to construct portfolios
        
class Factor_Model(object):
    
    def __init__(self, return_data, factor_data, n, factors, rf):
        
        self.r_data = return_data # Initialize the input data
        self.f_data = factor_data
        self.n = n # Initialize the number of assets (features)
        self.factors = factors
        self.rf = rf # Initialize the risk-free rate
        
    def r_mean_vec(self): # Create the expected return vector
        
        expected_return_vec = np.transpose(np.mat(self.r_data.mean()))
        
        return expected_return_vec
    
    def ex_r_mat(self): # Create the excess return matrix
        
        excess_return_matrix = np.mat(self.r_data - self.r_data.mean())
        
        return excess_return_matrix
    
    def get_AlphaBeta(self): # Create the Alpha vector and Beta matrix
    
        alpha_vec = []
        beta_mat = []
        reg = LinearRegression()
        
        if self.factors == 1:
    
            for i in range(0,self.n):
        
                reg.fit(np.array(self.f_data.iloc[:, 0: self.factors]).reshape(len(self.f_data), self.factors), 
                        np.array(self.r_data.iloc[:, i]).reshape(len(self.r_data), 1) - 
                        np.array(self.rf * np.ones((len(self.r_data),1))))
                
                alpha_vec.append(reg.intercept_)
                beta_mat.append(reg.coef_[0,0])
    
            alpha_vec = np.array(alpha_vec).reshape(self.n,1)
            beta_mat = np.array(beta_mat).reshape(self.n,1)
        
        else:
            
            for i in range(0,self.n):
        
                reg.fit(np.array(self.f_data.iloc[:, 0: self.factors]).reshape(len(self.f_data), self.factors), 
                        np.array(self.r_data.iloc[:, i]).reshape(len(self.r_data), 1) - 
                        np.array(self.rf * np.ones((len(self.r_data),1))))
                
                alpha_vec.append(reg.intercept_)
                beta_mat.append(reg.coef_[0,0: self.factors])
            
            alpha_vec = np.array(alpha_vec).reshape(self.n,1)
            beta_mat = np.mat(beta_mat)
                
        return alpha_vec, beta_mat
    
    def get_cov_mat(self): # Create the covariance matrix 
        
        _, beta = self.get_AlphaBeta()
        
        sd_assets = np.std(self.r_data)
        for i in range(self.factors):
            locals()['sd_factor'+str(i)] = []
        for i in range(self.factors):
            sd_factor = np.std(self.f_data.iloc[:, i])
            locals()['sd_factor'+str(i)].append(sd_factor)
            
        cov_mat_factors = np.zeros((self.factors, self.factors))
        for i in range(self.factors):
            cov_mat_factors[i, i] = [x**2 for x in locals()['sd_factor'+str(i)]][0]
            
        rows = []
        columns = []
        for i in range(self.n):
            random_string = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            rows.append(random_string)
            columns.append(random_string)           
        results = pd.DataFrame(0.0, columns = columns, index = rows)
        
        for a in range(0, self.n):
    
            for b in range(0, self.n):
    
                if columns[a] == rows[b]:
          
                    results.iloc[a, b] = sd_assets[a]**2
        
                else:
            
                    results.iloc[a, b] = np.dot(np.dot(beta[a,:], cov_mat_factors), np.transpose(beta)[:,b])
                
        results = np.mat(results)
        
        return results
    
    def num_abcd(self): # Calculate the numbers of a, b, c, d
        
        ones_vec = np.ones((self.n, 1))
        cov_matrix = self.get_cov_mat()
        expected_return_vec = self.r_mean_vec()
        
        a = ones_vec.T * (cov_matrix.I) * ones_vec
        b = ones_vec.T * (cov_matrix.I) * expected_return_vec
        c = expected_return_vec.T * (cov_matrix.I) * expected_return_vec
        d = (a*c - b**2)
        
        return a, b, c, d
            
    def tp_x_vec(self): # Calculate the X vector, the expected return and standard deviation of the tangency portfolio 
        
        alpha, beta = self.get_AlphaBeta()
        cov_mat = self.get_cov_mat()
        a, b, _, _ = self.num_abcd()
        
        risk_aversion = b - self.rf * a 
        cov_mat = self.get_cov_mat()
        r_mean = self.r_mean_vec()
        
        mu_factors = np.mat((self.f_data.iloc[:, 0:self.factors]).mean()).T
    
        x_vec = np.multiply(1/risk_aversion, np.dot(cov_mat.I, r_mean - 
                                                    (np.zeros((self.n,1)))))
        expected_return_tp = np.dot(x_vec.T, np.add(alpha, np.dot(beta, mu_factors)))
        sd_tp = np.sqrt(np.dot(np.dot(x_vec.T, cov_mat), x_vec))
    
        return x_vec, expected_return_tp, sd_tp
    
    def gmvp_x_vec(self): # Calculate the X vector, the expected return and standard deviation at the GMVP point
        
        a, b, _, _ = self.num_abcd()
        x_vec = np.dot(np.multiply(1/a, self.get_cov_mat().I), np.ones((self.n, 1)))
        expected_return_GMVP = np.array((b/a))
        sd_GMVP = np.array(np.sqrt(1/a))
    
        return x_vec, expected_return_GMVP, sd_GMVP
    
        
# 3. Define the Black-Litterman model to construct portfolios
        
class Black_Litterman_Model(object):
    
    def __init__(self, return_data, factor_data, MktCap_data, n, factors, rf):
        
        self.r_data = return_data # Initialize the input data
        self.f_data = factor_data # Initialize the factor data
        self.m_data = MktCap_data # Initialize the market capitalisation data
        self.n = n # Initialize the number of assets (features)
        self.factors = factors
        self.rf = rf # Initialize the risk-free rate   
    
    def get_MktCap(self): # Calculate the market capitalisation vector
        
        x_mktcap = []
        
        for i in self.m_data:   
            total = sum(self.m_data)
            weight = i/total
            x_mktcap.append(weight)
    
        return x_mktcap
    
    def get_Amkt(self): # Calculate the coefficient of risk aversion of the market portfolio
        
        market_return = (self.f_data['Mkt-RF'] + self.f_data['RF']).values
        mkt_var = (np.std(market_return, ddof = 1))**2
    
        A_market = np.average(self.f_data['Mkt-RF']) / mkt_var
    
        return A_market
    
    def get_cov_mat(self): # Calculate the covariance matrix 
    
        ex_r_mat = np.mat(self.r_data - self.r_data.mean())
        cov_matrix = np.mat((1 / (len(ex_r_mat) - 1)) * np.dot(ex_r_mat.T, ex_r_mat))
    
        return cov_matrix
    
    def get_pi(self): # Calculate the equillibrium return
    
        am = self.get_Amkt()   
        cov_mat = self.get_cov_mat()   
        xm = np.array(self.get_MktCap())
    
        pi = np.array(am * np.dot(cov_mat, xm))
    
        return pi.reshape(self.n,1)
    
    def get_Return(self, omega_matrix, p_matrix, q_vector, tau): # Calculate the expected return vector under the Black-Litterman Model
    
        cov_mat = self.get_cov_mat()
        pi_vec = self.get_pi()       
    
        first_part = ((tau*cov_mat).I + np.dot(np.dot(p_matrix.T, omega_matrix.I), p_matrix)).I
        second_part = np.dot((tau*cov_mat).I, pi_vec) + np.dot(np.dot(p_matrix.T, omega_matrix.I), q_vector)
    
        done = np.dot(first_part, second_part)
    
        return done
    
    def get_x_bl(self, Return_BL): # Calculate the weight allocation vector 
    
        am = self.get_Amkt()
        cov_mat = self.get_cov_mat()
    
        rbl = Return_BL - (np.average(self.f_data['RF'])*np.ones((len(Return_BL), 1)))
        x = ((am**(-1)) * np.dot(cov_mat.I, rbl)).tolist()
    
        revised_x = []   
        for i in x:       
            x_new = i / np.sum(x)
            revised_x.append(x_new)
        
        revised_x = np.array(revised_x)
    
        return revised_x
    
# 4. Define the function to derive the standard error of intercept
    
def lsqfity(X, Y):
    """
    ***Copy from StackFlow***
    
    Calculate a "MODEL-1" least squares fit.

    The line is fit by MINIMIZING the residuals in Y only.

    The equation of the line is:     Y = my * X + by.

    Equations are from Bevington & Robinson (1992)
    Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
    pp: 104, 108-109, 199.

    Data are input and output as follows:

    my, by, ry, smy, sby = lsqfity(X,Y)
    X     =    x data (vector)
    Y     =    y data (vector)
    my    =    slope
    by    =    y-intercept
    ry    =    correlation coefficient
    smy   =    standard deviation of the slope
    sby   =    standard deviation of the y-intercept

    """

    X, Y = map(np.asanyarray, (X, Y))

    # Determine the size of the vector.
    n = len(X)

    # Calculate the sums.

    Sx = np.sum(X)
    Sy = np.sum(Y)
    Sx2 = np.sum(X ** 2)
    Sxy = np.sum(X * Y)
    Sy2 = np.sum(Y ** 2)

    # Calculate re-used expressions.
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2

    # Calculate my, by, ry, s2, smy and sby.
    my = num / den
    by = (Sx2 * Sy - Sx * Sxy) / den
    ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

    diff = Y - by - my * X

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)
    sby = np.sqrt(Sx2 * s2 / den)

    return my, by, ry, smy, sby        