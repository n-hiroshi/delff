import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv_data(file_path):
    data = []
    # Open the CSV file and read the data
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(num) for num in row])

    return np.array(data)

def display_heatmap(data, xlabel, ylabel):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # Add axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

def main():
    file_path = 'data.csv' # Specify the path of the CSV file
    data = read_csv_data(file_path)

    xlabel = 'X Axis Label' # Set the label for the x-axis
    ylabel = 'Y Axis Label' # Set the label for the y-axis

    if data.shape == (100, 100):
        display_heatmap(data, xlabel, ylabel)
    else:
        print(f"Error: The CSV file should contain 100x100 data, but the actual shape is {data.shape}")

if __name__ == '__main__':
    main()

