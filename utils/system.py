import gin


class Runner:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            self.f()
            gin.clear_config()
        except Exception as e:
            print(e)
            gin.clear_config()
            raise e


def check_dataset_config(name: str):
    assert name in ['mnist', 'fashion_mnist', 'cifar10'], f'Invalid dataset name: {name}'
