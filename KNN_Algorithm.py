import statistics
import math


# in general, if you need index, just use enumerate
# otherwise, just use for each loops or items
# if u have to use loops, just use range

def knn(data, query, k, choice):
    if k > len(data):
        return "Incorrect K value"

    distance = []  # stores a tuple that contains index and distance
    for index, example in enumerate(data):
        currDist = euclideanDistance(query, example[0])
        distance.append((index, currDist))

    distance.sort(key=lambda x: x[1])  # set to sort based on the second element of the sublist

    neighbors = []

    # k Nearest Neighbors
    for i in range(k):
        neighbors.append(distance[i])

    ## regression data
    if choice == 0:
        total = 0
        for neighbor in neighbors:
            currExample = data[neighbor[0]]
            total += currExample[1]
        return total / len(neighbors)

    ## classification data
    if choice == 1:
        clf_neighbor_val = []
        for neighbor in neighbors:
            currExample = data[neighbor[0]]
            clf_neighbor_val.append(currExample[1])

        return statistics.mode(clf_neighbor_val)


def euclideanDistance(x1, x2) -> float:
    return math.sqrt(math.pow(x1 - x2, 2))


def main():
    # Column 0: height (inches)
    # Column 1: Weight (pounds)
    reg_data = [
        [65.75, 112.99],
        [71.52, 136.49],
        [69.40, 153.03],
        [68.22, 142.34],
        [67.79, 144.30],
        [68.70, 123.30],
        [69.80, 141.49],
        [70.01, 136.46],
        [67.90, 112.37],
        [66.49, 127.45],
    ]

    # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
    reg_query = 60
    print(knn(reg_data, reg_query, 3, 0))

    ## Column 0: age
    ## Column 1: 1 -> likes pizza, 0 -> doesn't like pizza
    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
    ]
    # Question:
    # Given the data we have, does a 33 year old like pineapples on their pizza?
    age = 20
    print(knn(clf_data, age, 3, 1))


if __name__ == "__main__":
    main()
