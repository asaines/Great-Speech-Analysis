#!/usr/bin/env python
# coding: utf-8
#https://www.python-graph-gallery.com/web-radar-chart-with-matplotlib
# In[11]:


import pyreadr
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from textwrap import wrap


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from palmerpenguins import load_penguins
import matplotlib.font_manager


# In[14]:


def radar(df):
    df_radar = (
        df.groupby('label').agg(
            Anger = ("anger", np.median),
            Disgust = ("disgust", np.median),
            Fear = ("fear", np.median),
            Joy = ("joy", np.median),
            Neutral = ('neutral', np.median),
            Sadness = ("sadness", np.median),
            Surprise = ("surprise", np.median),
            Entities_proportion= ("entities_proportion_in_speech", np.median),
            Imagery_proportion = ("imagery_proportion_in_speech", np.median),
        Stopwords_proportion= ("stopwords_proportion_in_speech", np.median),

        ))

    def rescale(x):
        return (x - np.min(x)) / np.ptp(x)

    df["re_mean_sentence_lengths"]=rescale(df["mean_sentence_lengths"])
    df["re_complexity"]=rescale(df["complexity"])

    df_radar2 = (
        df.groupby('label').agg(
            Sentence_lengths = ("re_mean_sentence_lengths", np.median),
            Complexity = ("re_complexity", np.median)).reset_index())

    df_radar=df_radar.merge(df_radar2, on="label")

    BG_WHITE = "#fbf9f4"
    BLUE = "#2a475e"
    GREY70 = "#b3b3b3"
    GREY_LIGHT = "#f2efe8"
    COLORS = ["#FF5A5F", "#FFB400", "#007A87"]


    # The types of speeches
    LABELS = df_radar["label"].values.tolist()

    # The four variables in the plot
    VARIABLES = df_radar.columns.tolist()[1:]
    VARIABLES_N = len(VARIABLES)

    # The angles at which the values of the numeric variables are placed
    ANGLES = [n / VARIABLES_N * 2 * np.pi for n in range(VARIABLES_N)]
    ANGLES += ANGLES[:1]

    # Padding used to customize the location of the tick labels
    X_VERTICAL_TICK_PADDING = 5
    X_HORIZONTAL_TICK_PADDING = 10    

    # Angle values going from 0 to 2*pi
    HANGLES = np.linspace(0, 2 * np.pi)

    # Used for the equivalent of horizontal lines in cartesian coordinates plots 
    # The last one is also used to add a fill which acts a background color.
    H0 = np.zeros(len(HANGLES))
    H1 = np.ones(len(HANGLES)) * 0.5
    H2 = np.ones(len(HANGLES))

    # Initialize layout ----------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, polar=True)

    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # Rotate the "" 0 degrees on top. 
    # There it where the first variable, avg_bill_length, will go.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Setting lower limit to negative value reduces overlap
    # for values that are 0 (the minimums)
    ax.set_ylim(-0.1, .65)

    # Plot lines and dots --------------------------------------------
    for idx, label in enumerate(LABELS):
        values = df_radar.iloc[idx].drop("label").values.tolist()
        values += values[:1]
        ax.plot(ANGLES, values, c=COLORS[idx], linewidth=4, label=label)
        ax.scatter(ANGLES, values, s=160, c=COLORS[idx], zorder=10)

    # Set values for the angular axis (x)
    ax.set_xticks(ANGLES[:-1])
    ax.set_xticklabels(VARIABLES, size=10)

    # Remove lines for radial axis (y)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")

    # Add custom lines for radial axis (y) at 0, 0.5 and 1.
    ax.plot(HANGLES, H0, ls=(0, (6, 6)), c=GREY70)
    ax.plot(HANGLES, H1, ls=(0, (6, 6)), c=COLORS[2])
    ax.plot(HANGLES, H2, ls=(0, (6, 6)), c=GREY70)

    # Now fill the area of the circle with radius 1.
    # This create the effect of gray background.
    ax.fill(HANGLES, H2, GREY_LIGHT)

    # Custom guides for angular axis (x).
    # These four lines do not cross the y = 0 value, so they go from 
    # the innermost circle, to the outermost circle with radius 1.
    ax.plot([0, 0], [0, 1], lw=2, c=GREY70)
    ax.plot([np.pi, np.pi], [0, 1], lw=2, c=GREY70)
    ax.plot([np.pi / 2, np.pi / 2], [0, 1], lw=2, c=GREY70)
    ax.plot([-np.pi / 2, -np.pi / 2], [0, 1], lw=2, c=GREY70)

    # Add levels -----------------------------------------------------
    # These labels indicate the values of the radial axis
    PAD = 0.05
    ax.text(-0.4, 0 + PAD, "0%", size=16, fontname="Roboto")
    ax.text(-0.4, 0.5 + PAD, "50%", size=16, fontname="Roboto")

    # Create and add legends -----------------------------------------
    # Legends are made from scratch.

    # Iterate through labels names and colors.
    # These handles contain both markers and lines.
    handles = [
        Line2D(
            [], [], 
            c=color, 
            lw=3, 
            marker="o", 
            markersize=8, 
            label=label
        )
        for label, color in zip(LABELS, COLORS)
    ]

    legend = ax.legend(
        handles=handles,
        loc=(1, 0),       # bottom-right
        labelspacing=1.5, # add space between labels
        frameon=False     # don't put a frame
    )

    # Iterate through text elements and change their properties
    for text in legend.get_texts():
        text.set_fontname("Roboto") # Change default font 
        text.set_fontsize(16)       # Change default font size

    # Adjust tick label positions ------------------------------------
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS[0::2]:
        tick.set_pad(X_VERTICAL_TICK_PADDING)

    for tick in XTICKS[1::2]:
        tick.set_pad(X_HORIZONTAL_TICK_PADDING)

    # Add title ------------------------------------------------------
    fig.suptitle(
        "Radar Plot of Speeches",
        x = 0.1,
        y = 1,
        ha="left",
        fontsize=32,
        fontname="Lobster Two",
        color=BLUE,
        weight="bold",    
    )
    return 

