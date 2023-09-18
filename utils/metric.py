import threading
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


class ImgMetric(object):
    """Evaluation Metrics for Image Classification"""

    def __init__(self, num_class=2, score_thresh=0.5):
        super(ImgMetric, self).__init__()
        self.num_class = num_class
        self.score_thresh = score_thresh
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.tol_label = np.array([], dtype=np.float64)
        self.tol_pred = np.array([], dtype=np.float64)

    def update_1(self, preds, labels):
        for pred, label in zip(preds, labels):
            self.tol_pred = np.append(self.tol_pred, pred)
            self.tol_label = np.append(self.tol_label, label)

    def update(self, preds, labels):
        def evaluate_worker(self, pred, label):
            with self.lock:
                self.tol_pred = np.append(self.tol_pred, pred)
                self.tol_label = np.append(self.tol_label, label)
            return

        if isinstance(preds, np.ndarray):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, pred, label),
                                        )
                       for (pred, label) in zip(preds, labels)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented

    def total_score(self):
        tol_auc = roc_auc_score(self.tol_label, self.tol_pred)

        tol_acc = accuracy_score(self.tol_label, np.where(self.tol_pred > self.score_thresh, 1, 0))
        
        fpr, tpr, _ = roc_curve(self.tol_label, self.tol_pred, pos_label=1)
        tol_eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        tn, fp, fn, tp = confusion_matrix(self.tol_label, np.where(self.tol_pred > self.score_thresh, 1, 0)).ravel()
        tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2

        return tol_acc, tol_auc, tol_f1, tol_eer


class PixMetric(object):
    """Computes pix-level Acc mIoU, F1, and MCC metric scores
    refer to https://github.com/Tramac/awesome-semantic-segmentation-pytorch 
    and https://github.com/Tianfang-Zhang/AGPCNet
    and https://github.com/SegmentationBLWX/sssegmentation
    """
    def __init__(self, num_class=2):
        super(PixMetric, self).__init__()
        self.numClass = num_class
        self.lock = threading.Lock()
        self.reset()
    
    def reset(self):
        self.total_matrix = np.zeros((self.numClass, self.numClass))

    def BatchConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def update_1(self, imgPredicts, imgLabels):
        for imgPredict, imgLabel in zip(imgPredicts, imgLabels):
            assert imgPredict.shape == imgLabel.shape

            self.batch_matrix = self.BatchConfusionMatrix(imgPredict, imgLabel)
            self.total_matrix += self.batch_matrix

    def update(self, imgPredicts, imgLabels):
        def evaluate_worker(self, imgPredict, imgLabel):
            assert imgPredict.shape == imgLabel.shape
            self.batch_matrix = self.BatchConfusionMatrix(imgPredict, imgLabel)
            with self.lock:
                self.total_matrix += self.batch_matrix
            return

        if isinstance(imgPredicts, np.ndarray):
            evaluate_worker(self, imgPredicts, imgLabels)
        elif isinstance(imgPredicts, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, imgPredict, imgLabel),
                                        )
                       for (imgPredict, imgLabel) in zip(imgPredicts, imgLabels)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            raise NotImplemented
 
    def total_score(self):
        tp, fp, fn, tn = self.total_matrix.ravel()

        tol_acc = (tp+tn) / (tp+fp+fn+tn)

        tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2

        tol_mIoU = (tp/(fn+fp+tp) + tn/(fn+fp+tn)) / 2

        tol_mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

        return tol_acc, tol_mIoU, tol_f1, tol_mcc


class PixMetricTest(object):
    """Computes pix-level Acc mIoU, F1, and MCC metric scores
    refer to https://github.com/Tramac/awesome-semantic-segmentation-pytorch 
    and https://github.com/Tianfang-Zhang/AGPCNet
    and https://github.com/SegmentationBLWX/sssegmentation
    """
    def __init__(self, num_class=2):
        super(PixMetricTest, self).__init__()
        self.numClass = num_class

    def BatchConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

 
    def total_score(self, total_matrix):
        tp, fp, fn, tn = total_matrix.ravel()

        tol_acc = (tp+tn) / (tp+fp+fn+tn)

        tol_f1 = (2.0 * tp / (2.0 * tp + fn + fp) + 2.0 * tn / (2.0 * tn + fn + fp)) / 2

        tol_mIoU = (tp/(fn+fp+tp) + tn/(fn+fp+tn)) / 2

        tol_mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

        return tol_acc, tol_mIoU, tol_f1, tol_mcc