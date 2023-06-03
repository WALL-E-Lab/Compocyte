from test_data import generate_test_adata

if __name__ == '__main__':
    test_adata = generate_test_adata()
    print(test_adata.obs)
    print(test_adata.var)
    print(test_adata.X)
