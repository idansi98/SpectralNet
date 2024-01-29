import torch
import numpy as np
import matplotlib.pyplot as plt

from data import load_data

from spectralnet import Metrics
from spectralnet import SpectralNet


def main():
    x_train, y_train, x_test, y_test = load_data("3_spehres")
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)   
    # X = torch.cat([x_train, x_test])
    X = x_train
    if y_train is not None:
        # y = torch.cat([y_train, y_test])
        y = y_train
    else:
        y = None

    spectralnet = SpectralNet(
        n_clusters=3,
        should_use_ae=False,
        should_use_siamese=True,
        spectral_batch_size=500,
        spectral_epochs=100,
        spectral_is_local_scale=False,
        spectral_n_nbg=8,
        spectral_scale_k=2,
        spectral_lr=1e-2,
        siamese_epochs=10,
        ae_epochs=1,
        spectral_hiddens=[128, 128, 3],
    )
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        y = y.detach().cpu().numpy()
        acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=3)
        nmi_score = Metrics.nmi_score(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y)
    plt.show()

    x_train, y_train, x_test, y_test = load_data("3_spehres_2")
    # no predicct for test data, and plot the embeddings, but now the number of clusters is 4
    cluster_assignments = spectralnet.predict(x_test)
    embeddings = spectralnet.embeddings_
    #convert y_test to array for plt.scatter
    y_test = y_test.detach().cpu().numpy()
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y_test)
    plt.show()

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()
