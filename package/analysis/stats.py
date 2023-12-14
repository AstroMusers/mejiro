

def chi_square(observed, expected):
    return (((observed - expected) ** 2) / expected).sum().sum()
