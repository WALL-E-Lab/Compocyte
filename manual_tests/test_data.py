from scFlorist.core.tools import get_leaf_nodes


def get_hierarchy():
    hierarchy = {
        'Blood': {
            'L': {
                'TNK': {
                    'T': {},
                    'NKT': {},
                    'NK': {},
                    'ILC': {}
                },
                'BP': {},
                'M': {}
            },
            'NL': {}
        }
    }

    return hierarchy


def generate_test_adata():
    print(get_leaf_nodes(get_hierarchy()))
    return
