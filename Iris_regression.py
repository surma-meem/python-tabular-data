#! /usr/bin/env python3

# Linear regression of sepal length on petal length for each Iris species.
# Reads iris.csv, then for each species makes a scatter plot with a
# regression line on top and saves it as a PNG.

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def regress_and_plot(species_data, species_name, output_file):
    """Run linear regression for one species and save the plot."""
    # Get the two columns we want
    x = species_data.petal_length_cm
    y = species_data.sepal_length_cm

    # Run the regression
    regression = stats.linregress(x, y)
    slope = regression.slope
    intercept = regression.intercept

    # Print the results
    print("Results for " + species_name + ":")
    print("  slope     =", slope)
    print("  intercept =", intercept)
    print("  r-squared =", regression.rvalue ** 2)
    print("  p-value   =", regression.pvalue)

    # Make the plot
    plt.figure()
    plt.scatter(x, y, label = "Data")
    plt.plot(x, slope * x + intercept, color = "orange", label = "Fitted line")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(species_name)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def main():
    # Read the data
    dataframe = pd.read_csv("iris.csv")

    # Make a separate data frame for each species
    setosa = dataframe[dataframe.species == "Iris_setosa"]
    versicolor = dataframe[dataframe.species == "Iris_versicolor"]
    virginica = dataframe[dataframe.species == "Iris_virginica"]

    # Run the regression and make a plot for each one
    regress_and_plot(setosa, "Iris_setosa", "petal_v_sepal_length_regress_setosa.png")
    regress_and_plot(versicolor, "Iris_versicolor", "petal_v_sepal_length_regress_versicolor.png")
    regress_and_plot(virginica, "Iris_virginica", "petal_v_sepal_length_regress_virginica.png")


if __name__ == '__main__':
    main()
