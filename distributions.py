
import sys
import os

base_path = os.getcwd()

"""
    Utility function for inverse normal distribution.
"""
def qnorm( p, mean = 0.0, sd = 1.0):
    if p <= 0 or p >= 1:
        raise ValueError("Argument to ltqnorm %f must be in interval (0,1)"%p)
 
    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)
 
    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow
 
    # Rational approximation for lower region:
    if p < plow:
       q  = math.sqrt(-2*math.log(p))
       z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
 
    # Rational approximation for upper region:
    elif phigh < p:
       q  = math.sqrt(-2*math.log(1-p))
       z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
 
    # Rational approximation for central region:
    else:
       q = p - 0.5
       r = q*q
       z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    
    return mean + z * sd

"""
    Globals
"""
mu1 = 1.25
sigma = 0.4
expected_prototype = 0.9
expected_consistent_a = 0.72
expected_consistent_b = 0.26
expected_inconsistent = 0.095

"""
    Compute and output distribution parameters.
"""
threshold = mu1 + sigma * qnorm(1 - expected_prototype)
mu2 = threshold - sigma * qnorm(1 - expected_consistent_a)
mu3 = threshold - sigma * qnorm(1 - expected_consistent_b)
mu4 = threshold - sigma * qnorm(1 - expected_inconsistent)

path = base_path + os.sep + 'parameters.txt'
f = open(path, 'w')
f.write('mu1' + '=' + str(mu1) + '\n')
f.write('mu2' + '=' + str(mu2) + '\n')
f.write('mu3' + '=' + str(mu3) + '\n')
f.write('mu4' + '=' + str(mu4) + '\n')
f.write('sigma' + '=' + str(sigma) + '\n')
f.write('threshold' + '=' + str(threshold) + '\n')
f.write('dims' + '=' + str(128) + '\n')
f.write('categories' + '=' + str(8) + '\n')
f.write('max_similarity' + '=' + str(0.9) + '\n')
f.write('participants' + '=' + str(32) + '\n')
f.write('attention_pstc' + '=' + str(0.1) + '\n')
f.write('N_per_D' + '=' + str(25) + '\n')
f.write('transition_t' + '=' + str(0.1) + '\n')
f.write('presentation_t' + '=' + str(0.3) + '\n')
f.close()
    
print 'Distribution parameters successfully generated!'
