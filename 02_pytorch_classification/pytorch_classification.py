from sklearn import datasets


if __name__ == '__main__':
    # Make 1000 samples
    n_samples = 1000
    # Create Circles
    X, y = datasets.make_circles(n_samples, noise=0.03, random_state=42)

    print(len(X), len(y))