# conda install tqdm

from tqdm import tqdm

def decorator(func, totalSteps=10, desc='In Progress'):
    @wrap(func)
    def wrapper(*args, **kwargs):
        with tqdm(total=totalSteps, desc=desc) as progressBar:
            def updateProgress(progress):
                progressBar.update(progress - progressBar.n)
        func(*args, **kwargs)

        if 'updateProgress' in func.__code__.co_varnames:
            kwargs['updateProgress'] = updateProgress
        
        result = func(*args, **kwargs)
        progressBar.update(totalSteps - progressBar.n)
        return result
    return wrapper


for i in tqdm(range(100), desc='progress', unit='item'):
    time.sleep(10)