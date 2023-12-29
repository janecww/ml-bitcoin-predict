import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D


def generate_simulation(test, pred, model_name, threshold, pred_scale=1, skip_plot=False):
    # Scale pred if necessary
    pred *= pred_scale

    ## SWING TRADING SIMULATION
    money = 1_000_000
    bitcoin = 0
    bc_price = 23936.3  # bc price at start of test period

    # Values here gotten from Preprocessing notebook, where we scaled the
    # target feature (and retrieved relevant mean/std)
    u, s = 14.6162963, 619597.97351221**0.5
    inverse_std_scale = lambda z: (z * s) - u
    np_inverse_std = np.vectorize(inverse_std_scale)
    test_unscaled = np_inverse_std(test)

    # Simulation logic
    for i, p in enumerate(pred):
        if p > threshold and money >= bc_price:
            # if our prediction is highly positive, purchase 1 bitcoin!
            bitcoin += 1
            money -= bc_price
        elif p < 0:
            # if our prediction is negative, sell all bitcoins!
            # 5% transaction fee
            money += bitcoin * bc_price * 0.98
            bitcoin = 0
        # bitcoin price changes at end of "day"
        bc_price += test_unscaled[i]

    final_assets = money + (bitcoin * bc_price)
    profit = final_assets - 1_000_000
    profit_str = "-" * (1 - (float(profit) > 0)) + "$" + str(round(abs(profit), 2))
    #print(f"Threshold: {threshold}, Final profits: {profit_str}")

    if skip_plot:
        return profit

    ## PLOTTING TEST/PRED AND PROFITABILITY
    red = (1, 0, 0, 1)
    green = (0, 0.8, 0, 1)
    yellow = (0.8, 0.8, 0, 1)
    grey = (0.4, 0.4, 0.4, 1)

    def check_correctness(test, pred, threshold):
        if test < 0 and pred >= threshold:
            return red    # we lose money!
        elif test > 0 and pred < threshold:
            return yellow # lost opportunity to make money
        elif test < 0 and pred < threshold:
            return grey   # did not buy (correct)
        else:
            return green  # did buy (correct)

    correctness = [check_correctness(t, p, threshold) for t, p in zip(test, pred)]

    # Visualize actual vs. predicted prices
    plt.figure(figsize=(15, 7))
    plt.plot(test.index, test.values, label="Actual", color="blue")
    plt.plot(test.index, pred, label="Predicted", color="red", linestyle="--")

    # Visualize swing trading profitability
    plt.scatter(test.index, test.values, s=100, color=correctness)
    custom_lines = [
        Line2D([0], [0], marker="o", markerfacecolor=green, color="w", markersize=10),
        Line2D([0], [0], marker="o", markerfacecolor=yellow, color="w", markersize=10),
        Line2D([0], [0], marker="o", markerfacecolor=red, color="w", markersize=10),
        Line2D([0], [0], marker="o", markerfacecolor=grey, color="w", markersize=10),
    ]
    custom_legend = plt.legend(
        custom_lines,
        ["Made profit", "Missed profit", "Made losses", "Avoided losses"],
        loc="upper right",
        title="Profitability",
        title_fontproperties={"weight": 1000, "size": "large"},
    )
    plt.gca().add_artist(custom_legend)  # persist this custom legend so we can have 2

    # Helper lines
    line_x_coords = [test.index[0] - 5, test.index[-1] + 5]
    plt.plot(  # THRESHOLD LINE
        line_x_coords,
        [threshold, threshold],
        label="Threshold",
        color="black",
        linestyle=":",
    )
    plt.plot(  # ZERO LINE
        line_x_coords,
        [0, 0],
        label="Zero",
        color="black",
    )

    # Other elements of plot
    plt.title(f"Actual vs. Predicted Bitcoin Price Changes ({model_name})")
    plt.text(
        682,  # right end of graph
        3.25,  # near top, below legend
        f"Swing trade profits: {profit_str}",
        backgroundcolor=(0.4, 0.4, 0.4, 0.5),  # transparent grey
        ha="right",
        fontstyle="oblique",
        size="large",
    )
    plt.xlabel("Index")
    plt.ylabel("Price Change (standardized)")
    plt.legend(
        title="Lines",
        title_fontproperties={"weight": 1000, "size": "large"},
        loc="upper left",
    )

    plt.grid(True)
    plt.tight_layout()

    # Saves chart as model_name_results.png and displays it too
    # Picture should be saved in same folder as this notebook (or script)
    plt.savefig(f"{model_name}_results.png")
    plt.show()


# Example usage (without scaling factor for predicted values)
# generate_simulation(y_test, y_pred, "GradientBoosted Regression", threshold=0.5)