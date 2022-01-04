import sys
import scipy as sp
import scipy.linalg as splin
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, SpatialDropout2D, GaussianNoise
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import Model, Sequential, layers, optimizers, activations, callbacks
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import io as sio
from skimage.transform.pyramids import pyramid_reduce
from skimage.transform import rescale

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py



class HSI:
    '''
    A class for Hyperspectral Image (HSI) data.
    '''
    def __init__(self, data, rows, cols, gt):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()
            
        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data,(self.rows,self.cols,self.bands))
        self.gt = gt
    
    def array(self):
        """this returns a array of spectra with shape num pixels x num bands
        
        Returns:
            a matrix -- array of spectra
        """
        return np.reshape(self.image,(self.rows*self.cols,self.bands))
    
    def get_bands(self, bands):
        return self.image[:,:,bands]

    def crop_image(self,start_x,start_y,delta_x=None,delta_y=None):
        if delta_x is None: delta_x = self.cols - start_x
        if delta_y is None: delta_y = self.rows - start_y
        self.cols = delta_x
        self.rows = delta_y
        self.image = self.image[start_x:delta_x+start_x,start_y:delta_y+start_y,:]
        return self.image

def load_HSI(path):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')
    
    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    
    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None
    
    return HSI(numpy_array, n_rows, n_cols, gt)

class SumToOne(layers.Layer):
    def __init__(self, params, **kwargs):
        super(SumToOne, self).__init__(**kwargs)
        self.num_outputs = params['num_endmembers']
        self.params = params
        
    def l_regularization(self,x):
        patch_size = self.params['patch_size']*self.params['patch_size']
        z = tf.abs(x + tf.keras.backend.epsilon())
        # l_half = tf.reduce_sum(tf.norm(z, self.params['l'], axis=3), axis=None)
        l_half = tf.reduce_sum(tf.norm(z, 1, axis=3), axis=None)
        return 1.0 / patch_size * self.params['l1'] * l_half
        
    def tv_regularization(self, x):
        patch_size = self.params['patch_size']*self.params['patch_size']
        #z = tf.abs(x + tf.keras.backend.epsilon())
        tv = tf.reduce_sum(tf.image.total_variation(x))
        return 1.0 / patch_size * self.params['tv'] * tv

    def call(self, x):
        if self.params['l1']>0.0:
            self.add_loss(self.l_regularization(x))
        if self.params['tv']>0.0:
            self.add_loss(self.tv_regularization(x))
        return x

class Scaling(layers.Layer):
    def __init__(self, params, **kwargs):
        super(Scaling, self).__init__(**kwargs)
        self.params = params
        
    def non_zero(self, x):
        patch_size = self.params['patch_size']*self.params['patch_size']
        #z = tf.abs(x + tf.keras.backend.epsilon())
        tv = tf.reduce_sum(tf.image.total_variation(x))
        return 1.0 / patch_size * self.params['tv'] * tv

    def call(self, x):
        self.add_loss(self.tv_regularization(x))
        return K.relu(x)

def SAD(y_true, y_pred):
    y_true2 = tf.math.l2_normalize(y_true, axis=-1)
    y_pred2 = tf.math.l2_normalize(y_pred, axis=-1)
    A = tf.keras.backend.mean(y_true2 * y_pred2)
    sad = tf.math.acos(A)
    return sad

def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos>1.0: cos = 1.0
    return np.arccos(cos)

def calcmSAD(Aorg,Ahat,startCol):
    Aorg = np.squeeze(Aorg)
    if Aorg.shape[1] > Aorg.shape[0]:
        Aorg = Aorg.T
    if Ahat.shape[1] > Ahat.shape[0]:
        Ahat = Ahat.T
    r1=np.min(Aorg.shape)
    r2=np.min(Ahat.shape)
    s = np.zeros((r1,r2))
    for i in range(r1):
        ao = Aorg[:,i]
        for j in range(r2):
            ah=Ahat[:,j]
            s[i,j]=np.min([SAD(ao,ah),SAD(ao,-ah)])
    s0=s
    sad = np.squeeze(np.zeros((1, r1)))
    idxHat = np.squeeze(np.zeros((1, r1)))
    idxOrg = np.squeeze(np.zeros((1, r1)))
    for p in range(r1):
        if startCol > -1 and p == 0:
            b = np.argmin(s[:, startCol])
            sad[p] = np.min(s[:, startCol])
            idxHat[p] = b
            idxOrg[p] = startCol
        else:
            sad[p]=np.min(s.flatten())
            (idxHat[p],idxOrg[p]) = np.unravel_index(np.argmin(s, axis=None), s.shape)
        s[:, int(idxOrg[p])] = np.inf
        s[int(idxHat[p]), :] = np.inf
        if np.isinf(sad[p]):
            idxHat[p] = np.inf
            idxOrg[p] = np.inf
    sad_k=sad
    a = np.sort(idxOrg).astype(int)
    b = np.argsort(idxOrg)
    idxHat=idxHat[b]
    sad_k=sad_k[b]
    sad_m=np.mean(sad_k)

    return sad_m, idxOrg.astype(int), idxHat.astype(int),sad_k

def asam_and_order(Aorg, Ahat):
    if Aorg.shape[1]>Aorg.shape[0]:
        Aorg=Aorg.T
    if Ahat.shape[1]>Ahat.shape[0]:
        Ahat=Ahat.T
    r=Aorg.shape[1]
    idxOrg=np.zeros((r,r),dtype=np.int8)
    idxHat=np.zeros((r,r),dtype=np.int8)
  
    sad_m, idxOrg, idxHat, sad_k = calcmSAD(Aorg, Ahat,-1)
    
    return idxHat, sad_m


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, ASAM / float(num)

def plotEndmembers(endmembers):
    if len(endmembers.shape)>2 and endmembers.shape[1] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=0).mean(axis=0)
    else:
        endmembers = np.squeeze(endmembers)
    # endmembers = endmembers / endmembers.max()
    num_endmembers = np.min(endmembers.shape)
    fig = plt.figure(num=1, figsize=(8, 8))
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[i, :], 'r', linewidth=1.0)
        ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('endm.png')
    plt.close()


def plotEndmembersAndGT(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
        
    hat, sad = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "mSAD: " + format(sad, '.3f') + " radians"
    st = plt.suptitle(title)
    
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'k', linewidth=1.0)
        ax.set_title(format(numpy_SAD(endmembers[hat[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

def plotAbundancesSimple(abundances,name):
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    n = num_endmembers / 2
    if num_endmembers % 2 != 0: n = n + 1
    cmap = 'viridis'
    plt.figure(figsize=[12, 12])
    AA = np.sum(abundances, axis=-1)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    #plt.savefig(name+'.png')
    plt.close()


class PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array()
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endmembersGT = hsi.gt
        self.sads = None
        self.epochs = []


    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.sads = []
    def get_re(self):
        return (self.epochs,self.RE)

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('SAD'))
        self.num_epochs = epoch
        endmembers = self.model.layers[-1].get_weights()[0]
        endmembers = np.squeeze(endmembers)


       
        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        if self.endmembersGT is not None:
            plotEndmembersAndGT(self.endmembersGT, endmembers)
        else:
            plotEndmembers(endmembers)
       

def reconstruct(A,S):
    s_shape = S.shape
    S = np.reshape(S,(S.shape[0]*S.shape[1],S.shape[2]))
    reconstructed = np.matmul(S,A)
    reconstructed = np.reshape(reconstructed, (s_shape[0], s_shape[1],reconstructed.shape[1]))
    return reconstructed


def estimate_snr(Y, r_m, x):

    # L number of bands (channels), N number of pixels
    [L, N] = Y.shape
    [p, N] = x.shape           # p number of endmembers (reduced dimension)

    P_y = sp.sum(Y**2)/float(N)
    P_x = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
    snr_est = 10*sp.log10((P_x - p/L*P_y)/(P_y - P_x))

    return snr_est


def vca(Y, R, verbose=True, snr_input=0):
    # Vertex Component Analysis
    #
    # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
    #
    # ------- Input variables -------------
    #  Y - matrix with dimensions L(channels) x N(pixels)
    #      each pixel is a linear mixture of R endmembers
    #      signatures Y = M x s, where s = gamma x alfa
    #      gamma is a illumination perturbation factor and
    #      alfa are the abundance fractions of each endmember.
    #  R - positive integer number of endmembers in the scene
    #
    # ------- Output variables -----------
    # Ae     - estimated mixing matrix (endmembers signatures)
    # indice - pixels that were chosen to be the most pure
    # Yp     - Data matrix Y projected.
    #
    # ------- Optional parameters---------
    # snr_input - (float) signal to noise ratio (dB)
    # v         - [True | False]
    # ------------------------------------
    #
    # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
    # This code is a translation of a matlab code provided by
    # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
    #
    # more details on:
    # Jose M. P. Nascimento and Jose M. B. Dias
    # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
    # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
    #
    #

    #############################################
    # Initializations
    #############################################
    if len(Y.shape) != 2:
        sys.exit(
            'Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape   # L number of bands (channels), N number of pixels

    R = int(R)
    if (R < 0 or R > L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

    #############################################
    # SNR Estimates
    #############################################

    if snr_input == 0:
        y_m = sp.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m           # data with zero-mean
        # computes the R-projection matrix
        Ud = splin.svd(sp.dot(Y_o, Y_o.T)/float(N))[0][:, :R]
        # project the zero-mean data onto p-subspace
        x_p = sp.dot(Ud.T, Y_o)

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10*sp.log10(R)

    #############################################
    # Choosing Projective Projection or
    #          projection to p-1 subspace
    #############################################

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

            d = R-1
            if snr_input == 0:  # it means that the projection is already computed
                Ud = Ud[:, :d]
            else:
                y_m = sp.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m  # data with zero-mean

                # computes the p-projection matrix
                Ud = splin.svd(sp.dot(Y_o, Y_o.T)/float(N))[0][:, :d]
                # project thezeros mean data onto p-subspace
                x_p = sp.dot(Ud.T, Y_o)

            Yp = sp.dot(Ud, x_p[:d, :]) + y_m      # again in dimension L

            x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
            c = sp.amax(sp.sum(x**2, axis=0))**0.5
            y = sp.vstack((x, c*sp.ones((1, N))))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        # computes the p-projection matrix
        Ud = splin.svd(sp.dot(Y, Y.T)/float(N))[0][:, :d]

        x_p = sp.dot(Ud.T, Y)
        # again in dimension L (note that x_p has no null mean)
        Yp = sp.dot(Ud, x_p[:d, :])

        x = sp.dot(Ud.T, Y)
        u = sp.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / sp.dot(u.T, x)

    #############################################
    # VCA algorithm
    #############################################

    indice = sp.zeros((R), dtype=int)
    A = sp.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = sp.random.rand(R, 1)
        f = w - sp.dot(A, sp.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = sp.dot(f.T, y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:, i] = y[:, indice[i]]        # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp
