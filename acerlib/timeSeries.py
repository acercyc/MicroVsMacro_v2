# ============================================================================ #
#                          1.0 - Acer 2017/04/24 15:44                         #
# ============================================================================ #
import numpy as np


def timeSeriesBatchGen(d, wSize, iRow=None, isForward=1, nPredictTime=1):
    """
        
        :param d: 2D or 3D matrix, 1st dimention is time
        :param wSize: window size
        :param iRow: row index
        :param isForward: extract forward or backword
        :param nPredictTime: # of prediction row
        :return: 2 lists
        
        

        1.0 - Acer 2017/01/26 15:12
        2.0 - Acer 2017/04/24 15:47
        """

    if d.ndim == 2:
        if iRow is None:
            iRow = range(d.shape[0] - nPredictTime - wSize + 1)

        dcat = []
        prediction = []
        for i in iRow:
            if isForward:
                sheet = d[i:i+wSize, :]
                dcat.append(sheet)
                prediction.append(d[i + wSize:i + wSize + nPredictTime, :])
            else:
                sheet = d[i:i-wSize+1, :]
                dcat.append(sheet)
                prediction.append(d[i + 1:i + 1 + nPredictTime, :])
        dcat = np.array(dcat)
        prediction = np.array(prediction)
        return dcat, prediction

    elif d.ndim == 3:
        if iRow is None:
            iRow = range(d.shape[0] - nPredictTime - wSize + 1)

        dcat = []
        prediction = []
        for i in iRow:
            if isForward:
                sheet = d[i:i+wSize, :, :]
                dcat.append(sheet)
                prediction.append(d[i + wSize:i + wSize + nPredictTime, :, :])
            else:
                sheet = d[i:i-wSize+1, :, :]
                dcat.append(sheet)
                prediction.append(d[i + 1:i + 1 + nPredictTime, :, :])
        dcat = np.array(dcat)
        prediction = np.array(prediction)
        return dcat, prediction



