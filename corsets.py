import numpy as np

num_dims = 250

def compute_distance_to_closest_center(centers_vector, data_point):
    # centers_vector - 2d numpy array of shape (num_centers, num_dims)
    # data_poin - 1d numpy array of shape (num_dims)
    
    num_centers = centers_vector.shape[0]
    
    minimum_distance = float('inf')

    for i in range(num_centers):
        current_distance = np.linalg.norm(centers_vector[i] - data_point)**2
        print('current_distance = ' + str(current_distance))
        if current_distance < minimum_distance:
            minimum_distance = current_distance

    return minimum_distance


def mapper(key, value):
    # key: None
    # value: 2d numpy array of shape (num_samples, num_dims) 

    # idea: create a corset from the input data
    # step 1: D^2 sampling
    
    D = [] #initialize D as empty set



    yield 0, "value"  # all mappers should yield the same key


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(200, 250)

if __name__ == "__main__":

    # testing the compute_distance_to_closest_center function
    centers_vector = np.array([[1.,1.],[2.,2.]])
    data_point = np.array([4.,4.])
    distance_to_closest_center = compute_distance_to_closest_center(centers_vector, data_point)
    print(distance_to_closest_center)
