import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import cv2 
import os
from scipy.stats import skew, kurtosis
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
import math
import sklearn
from typing import *
from tqdm import tqdm

from src.common.registry import Registry
from src.common.utils import write_report
from src.tasks.base import BaseTask


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (64, int(64 * image.shape[0]/image.shape[1] )))
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def mixture(vects, k, num_clines = 100):
    clf = GaussianMixture(n_components=k, max_iter=10000, tol=1e-5)
    clf.fit(vects)
    
    minq = vects[:, 0].min() - vects[:, 0].mean()*0.25
    maxq = vects[:, 0].max() + vects[:, 0].mean()*0.25
    x = np.linspace(minq, maxq, num = 200)
    
    minq = vects[:, 1].min() - vects[:, 1].mean()*0.25
    maxq = vects[:, 1].max() + vects[:, 1].mean()*0.25
    y = np.linspace(minq, maxq, num = 200)
    X, Y = np.meshgrid(x, y)
    
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    
    CS = plt.contour(
        X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(1, 5, num = num_clines)
    )
    plt.colorbar(CS, shrink=0.8, extend="both")



def main(imgs, vects, k = 10):
    sklearn.decomposition.PCA()
    fig, ax = plt.subplots(figsize = (28, 12))
    imgs = [x for x in imgs if isinstance(x, np.ndarray)]
    vects = sklearn.decomposition.PCA(2).fit_transform(vects)

    for n_, (img1) in enumerate(imgs): 
        
        x, y = vects[n_][0], vects[n_][1]
        imscatter(x, y, img1, zoom= .7, ax=ax)
        ax.plot(x, y)
    mixture(vects, k)

@Registry.register_task
class VisualTask(BaseTask):
    """
    Base task runner.
    """
    name: str = "visualize"
    def run(self, *args) -> None:
        images = []
        for sample in tqdm(self.query_dataset):
            image = sample.image
            for pp in self.preprocessing:
                if type(image) is list:
                    output = []

                    for img in image:
                        output.append(pp.run(img))
                else:
                    output = [pp.run(image)]

                if "bb" in output[0]:
                    images_list = []
                    bb_list = output[0]["bb"]

                    for bb in bb_list:
                        images_list.append(image[bb[0]:bb[2], bb[1]:bb[3]])

                    if len(images_list) > 0:
                        images.extend(images_list)

        vects = self.extractors[0].run(images, tokenizer=self.tokenizer)["result"]
        main(images, vects, 8)
        plt.savefig(f"{self.output_dir}/visualization-{self.query_dataset.name}-{self.extractors[0].name}.png")

