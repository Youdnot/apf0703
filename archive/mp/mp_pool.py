import time
import math
from multiprocessing import Pool

if __name__ == '__main__':
    start = time.perf_counter()
    results1 = [math.factorial(x) for x in range(10000)]
    end = time.perf_counter()
    
    print(f"Time taken: {end - start} seconds")

    start = time.perf_counter()
    with Pool(5) as p:
        results2 = p.map(math.factorial, list(range(10000)))
    end = time.perf_counter()

    print(f"Time taken: {end - start} seconds")

    print(all(x == y for x, y in zip(results1, results2)))


# Output:
# Time taken: 9.46591458300827 seconds
# Time taken: 2.701897665974684 seconds
# True