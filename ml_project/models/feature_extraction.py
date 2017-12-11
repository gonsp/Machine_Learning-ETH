import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

n_elements = 6822
n_features = 18286


class CardiogramFeatureExtractor():
    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        X = X.reshape(n_elements, n_features)
        X_new = None

        for id in range(0, n_elements):
            features = self.extract_features(X[id])
            if X_new is None:
                X_new = np.zeros((n_elements, len(features)))

            X_new[id] = features

        return X_new

    def extract_features(self, x):
        x = np.trim_zeros(x)

        peaks_top, peaks_bottom = self.extract_peaks_filtered(x)
        exit(0)

        x_new = []
        x_new.append(self.extract_mean(x))
        x_new.append(self.extract_variance(x))
        x_new.append(self.extract_min(x))
        x_new.append(self.extract_max(x))
        # x_new.append(self.extract_period(x))
        # x_new.append(self.extract_max(x))
        return list(x_new)

    def extract_peaks_filtered(self, x):
        peaks_top = self.extract_peaks(x, True)
        peaks_bottom = self.extract_peaks(x, False)

        deltas_top = self.extract_deltas(x, peaks_top, peaks_bottom)
        deltas_bottom = self.extract_deltas(x, peaks_bottom, peaks_top)

        self.plot_peaks(x, False, np.array(peaks_top))
        self.plot_peaks(x, False, np.array(peaks_top))
        self.plot_peaks(x, True, np.array(peaks_bottom))

        return (peaks_top, peaks_bottom)

    def extract_peaks(self, x, top):
        peaks = self.detect_peaks(x, threshold=0, mph=None, mpd=0, edge='rising', kpsh=True, valley=not top, show=False)
        def check_peak(pos):

            incr = x[pos-1] < x[pos]

            for i in range(pos+1, len(x)):
                if x[i] != x[pos]:
                    break

            return incr == (x[pos] >= x[i])

        return list(filter(check_peak, peaks))

    def extract_deltas(self, x, a, b):
        deltas = []

        j = 0
        for i in range(0, len(a)):
            if b[0] >= a[i]:
                continue

            while b[j] <= a[i]:
                j += 1
                if j >= len(b):
                    break

            if j >= len(b):
                break

            center = x[a[i]]
            left = x[b[j-1]]
            right = x[b[j]]

            delta = (a[i], (abs(center-left), abs(center-right)))
            _, delta_left, delta_right = delta
            if delta_right <= delta_left*2 and delta_left <= delta_right*2:
                deltas.append(delta)

        return deltas

    def extract_mean(self, x):
        return np.mean(x)

    def extract_variance(self, x):
        return np.var(x)

    def extract_min(self, x):
        return np.min(x)

    def extract_max(self, x):
        return np.max(x)

    def detect_peaks(self, x, mph=None, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False, show=False, ax=None):

        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).

        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.

        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`
        
        The function can handle NaN's 

        See this IPython Notebook [1]_.

        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            self.plot_peaks(x, valley, ind)

        return ind

    def plot_peaks(self, x, valley, ind):
        """Plot results of the detect_peaks function, see its help."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
        else:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

            ax.plot(x, 'b', lw=1)
            if ind.size:
                label = 'valley' if valley else 'peak'
                label = label + 's' if ind.size > 1 else label
                ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                        label='%d %s' % (ind.size, label))
                ax.legend(loc='best', framealpha=.5, numpoints=1)
            ax.set_xlim(-.02*x.size, x.size*1.02-1)
            ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
            yrange = ymax - ymin if ymax > ymin else 1
            ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
            ax.set_xlabel('Data #', fontsize=14)
            ax.set_ylabel('Amplitude', fontsize=14)
            mode = 'Valley detection' if valley else 'Peak detection'
            # plt.grid()
            plt.show()


I = 176
J = 208
K = 176


class HistogramGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X)
        return self

    def transform(self, X, y=None):
        # Data Selection
        trainData = np.reshape(X, (-1, I, J, K))
        trainData = np.asarray(trainData)
        selectedData = trainData[:, 17:157, 20:190, 45:145]
        print("Selected Data Size: " + str(selectedData.shape))

        # Generate histograms for cubes (10*10*10 cubes) for Training
        print("Generating histograms...\n")
        histStack = []
        for i_sample in range(0, selectedData.shape[0]):
            print("Histogram of MRI:", i_sample)
            histSubStack = []
            for x in range(0, 10):
                for y in range(0, 10):
                    for z in range(0, 10):
                        cube = selectedData[i_sample,
                                            14*x:14*(x+1),
                                            17*y:17*(y+1),
                                            10*z:10*(z+1)]
                        hist, bin_edges = np.histogram(cube, bins=50)
                        hist = hist.reshape(-1, 1)
                        histSubStack = np.append(histSubStack, hist)
            histSubStack = histSubStack.reshape(-1, 1)
            if i_sample == 0:
                histStack = np.append(histStack, histSubStack).reshape(-1, 1)
            else:
                histStack = np.append(histStack, histSubStack, axis=1)
        histTrainData = histStack.T
        X_new = histTrainData

        return X_new


class MeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, box_size=5):
        self.box_size = box_size

    def fit(self, X, y=None):
        print("----------")
        print("Fitting")
        self.max_value = np.amax(X)
        print("Max intensity", self.max_value)
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, I, J, K)

        X_new = np.zeros(X.shape, dtype=np.uint16)

        for i in range(0, X.shape[0]):
            print("Computing mean matrix for element", i)
            X_new[i] = self.compute_mean_matrix_3D(X[i])
            print("Mean matrix computed")

        X_new = X_new.reshape(-1, I * J * K)
        return X_new

    def compute_mean_matrix_3D(self, X):
        X_new = np.zeros(X.shape, dtype=np.uint32)

        for i in range(0, X.shape[0]):
            self.compute_mean_matrix(X[i], X_new[i])

            aux1 = np.rot90(X_new[i], axes=(0, 1))
            aux2 = np.zeros(aux1.shape)

            self.compute_mean_matrix(aux1, aux2)

            X_new[i] = np.rot90(aux2, axes=(1, 0))

        aux1 = np.rot90(X_new, axes=(0, 2))
        aux2 = np.zeros(aux1.shape)

        for i in range(0, aux1.shape[0]):
            self.compute_mean_matrix(aux1[i], aux2[i])

        X_new[:] = np.rot90(aux2, axes=(2, 0))

        for i in range(0, I):
            for j in range(0, J):
                for k in range(0, K):
                    length_x = \
                        min(I - 1, i + self.box_size) - \
                        max(0, i - self.box_size) + 1
                    length_y = \
                        min(J - 1, j + self.box_size) - \
                        max(0, j - self.box_size) + 1
                    length_z = \
                        min(K - 1, k + self.box_size) - \
                        max(0, k - self.box_size) + 1
                    X_new[i][j][k] /= length_x * length_y * length_z
                    # compute the mean
                    # X_new[i][j][k] /= self.max_value # normalize

        return X_new.astype(np.uint16)

    def compute_mean_matrix(self, X, M):
        for i in range(0, M.shape[0]):
            s = np.uint32(0)
            for j in range(0, self.box_size + 1):
                s += X[i][j]
            M[i][0] = s

            for j in range(1, M.shape[1]):
                old_pos = j - self.box_size - 1
                if old_pos >= 0:
                    s -= X[i][old_pos]
                new_pos = j + self.box_size
                if new_pos < M.shape[1]:
                    s += X[i][new_pos]
                M[i][j] = s
