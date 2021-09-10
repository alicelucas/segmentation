import matplotlib.pyplot as plt
import numpy as np

def datasize_vs_iou():
    """
    From our experiment we know what is the resulting IOU for each dataset size. We plot this infrmation
    """

    #TODO instead of writing this manually, you can parse each baseline-XX file and compute the number of images
    #used by doing XX * 670 (for DSB), where XX is the percentage.
    #You can read the y values from the jaccard.txt files.
    x = [16, 33, 67, 167, 335, 502, 670] #number of images
    y = [0.43, 0.47, 0.66, 0.71, 0.64, 0.72, 0.73] #the mean IoU score

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xticks(x)
    ax.set(xlabel="Number of images", ylabel="mean IoU", title="DSB")

    fig.savefig("plot.png")

