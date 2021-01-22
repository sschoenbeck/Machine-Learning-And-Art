import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras import layers


def main():
    path = os.path.dirname(os.path.dirname(os.path.normpath(__file__)))
    training_df = pd.read_csv(f'{path}/dataset_generator/timeline_data/2001_A_SPACE_ODYSSEY_complete_data.csv')
    print(training_df)
    plt.close("all")
    training_df.plot()
    plt.show()


if __name__ == "__main__":
    main()
