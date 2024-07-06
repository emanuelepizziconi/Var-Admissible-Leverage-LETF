import numpy as np
import matplotlib.pyplot as plt

beta = np.linspace(0, 5, 1000)
T = 1
mu = 0.1531
r = 0.048
f = 0.000945
sigma = 0.1834
betaa = 0.01
accrisk = 0.5
def arh (beta,C,alpha):
 b = np.abs(beta)*sigma*norm.ppf(alpha)
 ARH = ((-b/2-np.sqrt((b**2)/4+phi(beta)*np.log(1-C)))/phi(beta))**2
 nan_indices = np.isnan(ARH)
 ARH[nan_indices] = 100000
 return  ARH
def phi (beta):
 return (beta*(mu-r)+r-f*np.square(beta)-beta**2*sigma**2/2)
def rhat (beta):
 return np.exp((beta*(mu-r)+r-f*np.square(beta))*T)-1
def sigmahat (beta):
 return np.exp((beta*(mu-r)+r-f*np.square(beta))*T)*np.sqrt(np.exp(beta**2*sigma**2*T)-1)
def nromal (beta):
 return norm.cdf((np.log(1-z)-phi(beta)*T)/(np.abs(beta)*sigma*np.sqrt(T)))
def var (alfa,beta, T):
 return 1 - np.exp(phi(beta)*T+ np.abs(beta)*sigma*np.sqrt(T)*norm.ppf(alfa))
undert = 0
inrhat = 0
insigmahat = 0
inbeta = 0
while betaa<5:
    if (rhat(betaa)-r)/sigmahat(betaa)<accrisk:
        if undert == 0:
         undert = 1
         termrhat = rhat(betaa)
         termsigmahat = sigmahat(betaa)
         termbeta = betaa
    else:
        if undert == 1:
         undert = 0
         inrhat = rhat(betaa)
         insigmahat = sigmahat(betaa)
         inbeta = betaa
    betaa = betaa + 0.001

betab = np.linspace(inbeta, termbeta, 1000)
ax = plt.axes()
plt.plot(sigmahat(beta), rhat(beta)-r, color='black', linewidth =0.5)
plt.plot(sigmahat(betab), rhat(betab)-r, color='black')
ax.xaxis.set_ticks(np.arange(0, 2, 0.1))
ax.yaxis.set_ticks(np.arange(0, 2, 0.1))
plt.xlabel('Std. Deviation')
plt.ylabel('Expected Return - r')
plt.xlim(0,1)
plt.ylim(0,0.5)
plt.grid()
plt.show()
plt.savefig("admissible leverage1",dpi=1200)
from scipy.stats import norm
z =0.2
T = 2
beta = np.linspace(-5, 5, 1000)
print("Risk Free Rate: ",r,"Time: ",T," Years","Var: ",sigma,"Fees (yearly %): ",f,"Expected Return: ",mu, "Acceptable Risk (c): ", accrisk, "Max Acceptable Leverage: ", termbeta)
ax = plt.axes()
cumnorm = norm.cdf((np.log(1-z)-phi(beta)*T)/(np.abs(beta)*sigma*np.sqrt(T)))
plt.plot(beta, cumnorm, color='black')
z = 0.1
cumnorm = norm.cdf((np.log(1-z)-phi(beta)*T)/(np.abs(beta)*sigma*np.sqrt(T)))
plt.plot(beta, cumnorm, color='black',linestyle="dashed")
z = 0.005
cumnorm = norm.cdf((np.log(1-z)-phi(beta)*T)/(np.abs(beta)*sigma*np.sqrt(T)))
plt.plot(beta, cumnorm, color='black',linestyle="dotted")
ax.xaxis.set_ticks(np.arange(-5, 5.001, 1))
ax.yaxis.set_ticks(np.arange(0, 1.001, 0.1))
plt.xlabel('Leverage (Beta)')
plt.ylabel('Loss probability (T = 2)')
plt.xlim(-5,5.001)
plt.ylim(0,1.001)
plt.grid()
plt.legend(['z = 0.2', 'z = 0.1', 'z = 0.05'], loc='upper right')
plt.show()
plt.savefig("loss probability",dpi=1200)
accvar = 0.4
ax = plt.axes()
plt.plot(beta, var(0.05,beta,1), color='black')
plt.plot(beta, var(0.02,beta,1), color='black',linestyle="dashed")
plt.plot(beta, var(0.01,beta,1), color='black',linestyle="dotted")
ax.xaxis.set_ticks(np.arange(-5, 5.001, 1))
ax.yaxis.set_ticks(np.arange(-0.05, 1.001, 0.05))
plt.xlabel('Leverage (Beta)')
plt.ylabel('Value at Risk %')
plt.xlim(-5,5.001)
plt.ylim(-0.05,1.001)
plt.grid()
plt.legend(['alpha = 0.05', 'alpha = 0.02', 'alpha = 0.01'], loc='upper center')
plt.show()
var005 = var(0.05,beta,1)
var002 = var(0.02,beta,1)
var001 = var(0.01,beta,1)
var5 = var005[var005<accvar]
beta5 = beta[var005<accvar]
var2 = var002[var002<accvar]
beta2 = beta[var002<accvar]
var1 = var001[var001<accvar]
beta1 = beta[var001<accvar]
ax = plt.axes()
plt.plot(beta5, var5, color='black')
plt.plot(beta2, var2, color='black',linestyle="dashed")
plt.plot(beta1, var1, color='black',linestyle="dotted")
plt.plot(beta, var(0.05,beta,1), color='black',linewidth =0.5)
plt.plot(beta, var(0.02,beta,1), color='black',linestyle="dashed",linewidth =0.5)
plt.plot(beta, var(0.01,beta,1), color='black',linestyle="dotted",linewidth =0.5)
plt.axhline(y = accvar, color = 'black', linestyle = '-',linewidth =0.5)
ax.xaxis.set_ticks(np.arange(-5, 5.001, 1))
ax.yaxis.set_ticks(np.arange(-0.1, 1.001, 0.05))
plt.xlabel('Leverage (Beta)')
plt.ylabel('Value at Risk % (T=1)')
plt.xlim(-5,5.001)
plt.ylim(-0.1,1.001)
plt.grid()
plt.legend(['alpha = 0.05', 'alpha = 0.02', 'alpha = 0.01'], loc='upper center')
plt.show()
print("Investment Horizon (Years):", T, "Acceptable VaR: ", accvar)
print("Alpha = 0.05:","Min Acceptable Leverage: ",beta5[1],"Max Acceptable Leverage: ",beta5[-1])
print("Alpha = 0.02:","Min Acceptable Leverage: ",beta2[1],"Max Acceptable Leverage: ",beta2[-1])
print("Alpha = 0.01:","Min Acceptable Leverage: ",beta1[1],"Max Acceptable Leverage: ",beta1[-1])

ax = plt.axes()
plt.plot(beta, arh(beta,0.3,0.01), color='black')
plt.plot(beta, arh(beta,0.4,0.01), color='black',linestyle="dashed")
plt.plot(beta, arh(beta,0.5,0.01), color='black',linestyle="dotted")
ax.xaxis.set_ticks(np.arange(-5, 5.001, 1))
ax.yaxis.set_ticks(np.arange(0, 3.001, 0.5))
plt.xlabel('Leverage (Beta)')
plt.ylabel('Admissible Risk Horizon (alpha = 0.01)')
plt.xlim(-5,5.001)
plt.ylim(0,3.001)
plt.legend(['C = 0.3', 'C = 0.4', 'C = 0.5'], loc='upper left')
plt.grid()
plt.show()
ax = plt.axes()
plt.plot(beta, arh(beta,0.3,0.05), color='black')
plt.plot(beta, arh(beta,0.4,0.05), color='black',linestyle="dashed")
plt.plot(beta, arh(beta,0.5,0.05), color='black',linestyle="dotted")
ax.xaxis.set_ticks(np.arange(-5, 5.001, 1))
ax.yaxis.set_ticks(np.arange(0, 3.001, 0.5))
plt.xlabel('Leverage (Beta)')
plt.ylabel('Admissible Risk Horizon (alpha = 0.05)')
plt.xlim(-5,5.001)
plt.ylim(0,3.001)
plt.legend(['C = 0.3', 'C = 0.4', 'C = 0.5'], loc='upper left')
plt.grid()
plt.show()
