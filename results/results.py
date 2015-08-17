import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Stats functions
def sample(data):
    for x in data:
        yield random.choice(data)

def bootstrapci(data,func,n=3000,p=0.95):
    index=int(n*(1-p)/2)
    r=[func(list(sample(data))) for i in range(n)]
    r.sort()
    return r[index], r[-index]

def mean(x):
    return sum(x)/len(x)
    
def binomialci(p,N):
    data=[0]*N
    for i in range(int(N*p)):
        data[i]=1.0
    return bootstrapci(data,mean)

# Data loading functions 
def load_posner(dvals):
    files = []
    trend = np.zeros((len(dvals), 4))
    for dval in dvals:
        filename = 'Posner' + str(dval) + '.npy'
        files.append(filename)
    for f in files:
        data = np.load(f)
        trend[files.index(f), :] = data
    return trend

def load_brooks(dvals):
    files = []
    trend = np.zeros((len(dvals), 3))
    for dval in dvals:
        filename = 'Brooks' + str(dval) + '.npy'
        files.append(filename)
    for f in files:
        data = np.load(f)
        trend[files.index(f), :] = data
    return trend


def plot_posner_comp():
    N = 4
    data_avg = (0.13, 0.149, 0.269, 0.383)
    data_err =   (0.049, 0.068, 0.057, 0.068)
    m_data = load_posner(np.arange(0.05,0.16,0.01))

    # Compute MSE for each distortion val
    mse = np.zeros(len(m_data[:,0]))    
    temp = 0
    for i in range(len(m_data[:,0])):
        for j in range(len(m_data[0,:])):
            temp += np.square(m_data[i,j] - data_avg[j])
        mse[i] = temp
        temp = 0 

    min_dval = mse.argmin()

    temp1 = binomialci(m_data[min_dval,0], 6*32) 
    temp2 = binomialci(m_data[min_dval,1], 3*32) 
    temp3 = binomialci(m_data[min_dval,2], 6*32) 
    temp4 = binomialci(m_data[min_dval,3], 6*32)

    training_err = (temp1[0]-temp1[1]) / 2
    prototype_err = (temp2[0]-temp2[1]) / 2
    low_err = (temp3[0]-temp3[1]) / 2
    high_err = (temp4[0]-temp4[1]) / 2

    model_avg = (m_data[min_dval,0],m_data[min_dval,1],
                 m_data[min_dval,2],m_data[min_dval,3])
    model_err = (training_err, prototype_err, low_err, high_err)


    index = np.arange(N)
    width = 0.35     
    error_config = {'ecolor': '0.3'}

    fig, ax = plt.subplots(figsize=(14,8))

    data = plt.bar(index, data_avg, width,
                    alpha = 0.5,
                    color='red',
                    yerr=data_err,
                    capsize=7,
                    error_kw=error_config,
                    label='Data')

    model = plt.bar(index+width, model_avg, width,
                    alpha = 0.5,
                    color='blue',
                    yerr=model_err,
                    capsize=7,
                    error_kw=error_config,
                    label='Model')

    plt.xlabel('Stimulus Category', fontsize=25)
    plt.ylabel('Error', fontsize=25)
    plt.title('Experiment 1 - Model Evaluation', fontsize=30, y=1.02)
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=20
    ax.title.labelpad=35
    ax.set_ylim([0,0.6])
    plt.xticks(index + width, 
               ('Training', 'Prototype', 'Low Distortion', 'High Distortion'), 
               fontsize=20)
    plt.legend(fontsize='large')
    fig.savefig('posner-comp.png')

def plot_brooks_comp():
    N = 3
    data_avg1 = (0.32, 0.47, 0.33)
    data_err1 = (0.078, 0.125, 0.109)
    data_avg2 = (0.11, 0.19, 0.77)
    data_err2 = (0.055, 0.094, 0.102)

    m_data = load_brooks(np.arange(0.01,0.16,0.01))

    # Compute MSE for each distortion val
    mse1 = np.zeros(len(m_data[:,0]))
    mse2 = np.zeros(len(m_data[:,0]))
    temp1 = 0
    temp2 = 0
    for i in range(len(m_data[:,0])):
        for j in range(len(m_data[0,:])):
            temp1 += np.square(m_data[i,j] - data_avg1[j])
            temp2 += np.square(m_data[i,j] - data_avg2[j])
        mse1[i] = temp1
        mse2[i] = temp2
        temp1 = 0
        temp2 = 0 

    min_dval1 = mse1.argmin()
    min_dval2 = mse2.argmin()

    temp1 = binomialci(m_data[min_dval1,0], 16*8) 
    temp2 = binomialci(m_data[min_dval1,1], 16*4) 
    temp3 = binomialci(m_data[min_dval1,2], 16*4) 

    training_err = (temp1[0]-temp1[1]) / 2
    GT_err = (temp2[0]-temp2[1]) / 2
    BT_err = (temp3[0]-temp3[1]) / 2

    model_avg1 = (m_data[min_dval1,0],m_data[min_dval1,1],m_data[min_dval1,2])
    model_err1 = (training_err, GT_err, BT_err)

    temp1 = binomialci(m_data[min_dval2,0], 16*8) 
    temp2 = binomialci(m_data[min_dval2,1], 16*4) 
    temp3 = binomialci(m_data[min_dval2,2], 16*4) 

    training_err = (temp1[0]-temp1[1]) / 2
    GT_err = (temp2[0]-temp2[1]) / 2
    BT_err = (temp3[0]-temp3[1]) / 2

    model_avg2 = (m_data[min_dval2,0],m_data[min_dval2,1],m_data[min_dval2,2])
    model_err2 = (training_err, GT_err, BT_err)

    index = np.arange(N)
    width = 0.35     
    error_config = {'ecolor': '0.3'}

    fig, (ax1, ax2) = plt.subplots(1, 2, 
                                  sharey=True, 
                                  sharex=True,
                                  figsize=(24, 12))
    
    data1 = ax1.bar(index, data_avg1, width,
                    alpha = 0.5,
                    color='red',
                    yerr=data_err1,
                    capsize=7,
                    error_kw=error_config,
                    label='Data')

    model1 = ax1.bar(index+width, model_avg1, width,
                    alpha = 0.5,
                    color='blue',
                    yerr=model_err1,
                    error_kw=error_config,
                    label='Model')

    data2 = ax2.bar(index, data_avg2, width,
                    alpha = 0.5,
                    color='red',
                    yerr=data_err2,
                    error_kw=error_config,
                    label='Data')

    model2 = ax2.bar(index+width, model_avg2, width,
                    alpha = 0.5,
                    color='blue',
                    yerr=model_err2,
                    error_kw=error_config,
                    label='Model')

    ax1.set_ylabel('Error', fontsize=25)
    ax1.set_ylim([0,1])

    ax1.set_title('Composite Stimuli', fontsize=25, y=1.02)
    ax2.set_title('Individuated Stimuli', fontsize=25, y=1.02)

    ax1.set_xlabel('Stimulus Category', fontsize=25, labelpad=10)
    ax2.set_xlabel('Stimulus Category', fontsize=25, labelpad=10)

    ax1.set_xticks(index + width)
    ax2.set_xticks(index + width)

    plt.legend(fontsize='large', loc='upperleft')

    ax1.set_xticklabels(('Training', 'GT', 'BT'), fontsize=20)
    ax2.set_xticklabels(('Training', 'GT', 'BT'), fontsize=20)

    fig.suptitle('Experiment 2 - Model Evaluation', fontsize=40, y=1)
    fig.savefig('brooks-comp.png')

def plot_murphy_comp():
    N = 4
    model_avg = np.load('Murphy.npy')

    temp1 = binomialci(model_avg[0], 8*20) 
    temp2 = binomialci(model_avg[1], 8*20) 
    temp3 = binomialci(model_avg[2], 8*20) 
    temp4 = binomialci(model_avg[3], 8*20)

    prototype_err = (temp1[0]-temp1[1]) / 2
    consistent_err = (temp2[0]-temp2[1]) / 2
    inconsistent_err = (temp3[0]-temp3[1]) / 2
    control_err = (temp4[0]-temp4[1]) / 2

    model_err = (prototype_err, consistent_err, inconsistent_err, control_err)

    data_avg = (0.90, 0.72, 0.26, 0.095)
    data_err =   (0.02, 0.041, 0.039, 0.026)

    index = np.arange(N)
    width = 0.25     
    error_config = {'ecolor': '0.3'}
    fig, ax = plt.subplots(figsize=(14,8))
    data = plt.bar(index-width, data_avg, width,
                    alpha = 0.5,
                    color='red',
                    yerr=data_err,
                    capsize=7,
                    error_kw=error_config,
                    label='Data')

    model = plt.bar(index, model_avg, width,
                    alpha = 0.5,
                    color='blue',
                    yerr=model_err,
                    error_kw=error_config,
                    label='Model - Internal')

    plt.xlabel('Stimulus Category', fontsize=25)
    plt.ylabel('Positive Judgments', fontsize=25)
    plt.title('Experiment 3 - Model Evaluation', fontsize=30, y=1.04)

    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=20
    ax.title.labelpad=35
    ax.set_ylim([0,1])
    plt.xticks(index+width/2, 
               ('Prototype', 'Consistent', 'Inconsistent', 'Control'), 
               fontsize=20)
    plt.legend()
    fig.savefig('murphy-comp.png')

def plot_posner_trend():
    N = 4
    fig = plt.figure(figsize=(16,8.5))

    data = load_posner(np.arange(0.05,0.16,0.01))

    training_avg = data[:,0]
    prototype_avg = data[:,1]
    low_avg = data[:,2]
    high_avg = data[:,3]

    training_err = np.zeros(len(data[:,0]))
    prototype_err = np.zeros(len(data[:,0]))
    low_err = np.zeros(len(data[:,0]))
    high_err = np.zeros(len(data[:,0]))

    for x in range(len(data[:,0])):
        temp1 = binomialci(training_avg[x], 32*6) 
        temp2 = binomialci(prototype_avg[x], 32*3) 
        temp3 = binomialci(low_avg[x], 32*6) 
        temp4 = binomialci(high_avg[x], 32*6) 

        training_err[x] = (temp1[0]-temp1[1]) / 2
        prototype_err[x] = (temp2[0]-temp2[1]) / 2
        low_err[x] = (temp3[0]-temp3[1]) / 2
        high_err[x] = (temp4[0]-temp4[1]) / 2

    sigma = np.arange(0.05,0.16,0.01)

    plt.errorbar(sigma, training_avg, yerr=training_err, 
                 label='Training', color='green')
    plt.errorbar(sigma, prototype_avg, yerr=prototype_err, 
                 label='Prototype', color='blue')
    plt.errorbar(sigma, low_avg, yerr=low_err, 
                 label='Low Distortion', color='orange')
    plt.errorbar(sigma, high_avg, yerr=high_err, 
                 label='High Distortion', color='red')
    plt.xlabel('Sigma', fontsize='35', labelpad=10)
    plt.ylabel('Error', fontsize='35', labelpad=20)
    plt.ylim([0,0.85])
    plt.title('Experiment 1 - Prototype Categorization', fontsize=40, y=1.04)
    plt.legend(loc='upperleft')
    fig.savefig('posner-trend.png')

def plot_brooks_trend():
    N = 3
    fig = plt.figure(figsize=(16,8.5))
    
    data = load_brooks(np.arange(0.01,0.16,0.01))
    training_avg = data[:,0]
    GT_avg = data[:,1]
    BT_avg = data[:,2]

    training_err = np.zeros(len(data[:,0]))
    GT_err = np.zeros(len(data[:,0]))
    BT_err = np.zeros(len(data[:,0]))

    for x in range(len(data[:,0])):
        temp1 = binomialci(training_avg[x], 16*8) 
        temp2 = binomialci(GT_avg[x], 16*4) 
        temp3 = binomialci(BT_avg[x], 16*4) 

        training_err[x] = (temp1[0]-temp1[1]) / 2
        GT_err[x] = (temp2[0]-temp2[1]) / 2
        BT_err[x] = (temp3[0]-temp3[1]) / 2

    sigma = np.arange(0.01,0.16,0.01)

    plt.errorbar(sigma, training_avg, yerr=training_err, 
                label='Training', color='green')
    plt.errorbar(sigma, GT_avg, yerr=GT_err, 
                 label='Good Transfer', color='blue')
    plt.errorbar(sigma, BT_avg, yerr=BT_err, 
                 label='Bad Transfer', color='red')
    plt.xlabel('Sigma', fontsize='25', labelpad=10)
    plt.ylabel('Error', fontsize='25', labelpad=20)
    plt.ylim([0,1])
    plt.xlim([0,0.16])
    plt.title('Experiment 2 - Exemplar Categorization', fontsize=40, y=1.04)
    plt.legend(loc='upperleft')

    fig.savefig('brooks-trend.png')

plot_murphy_comp()
plot_brooks_trend()
plot_posner_trend()
plot_brooks_comp()
plot_posner_comp()