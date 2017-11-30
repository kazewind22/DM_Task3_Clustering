import numpy as np

num_dims = 250
k = 5 # number of centers

def compute_distance_to_closest_center(data_point, centers_vector):
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
    
    num_data_points = value.shape[0]

    # idea: create a corset from the input data
    # step 1: D^2 sampling
    
    D = np.zeros(k)
    
    #choose first cluster center uniformly at random
    B_indices[0] = np.random.randint(A.shape[0], size=1)
    #here I am not sure, for now choosing next point as point with highest "sensitivity" but the denomitor consists only of the points that in the 
    
    for center_index in range(1, k):
        for data_point_index in range(num_data_points):
            
        
    


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
    distance_to_closest_center = compute_distance_to_closest_center(data_point, centers_vector)
    print(distance_to_closest_center)
