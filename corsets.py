import numpy as np

np.random.seed(42)

num_dims = 250
k = 5 # number of centers

def compute_distance_to_closest_center(data_point, centers_vector):
    # centers_vector - 2d numpy array of shape (num_centers, num_dims)
    # data_poin - 1d numpy array of shape (num_dims)
    
    num_centers = centers_vector.shape[0]
    
    minimum_distance = float('inf')

    for i in range(num_centers):
        current_distance = np.linalg.norm(centers_vector[i] - data_point)**2
        if current_distance < minimum_distance:
            minimum_distance = current_distance

    return minimum_distance


def mapper(key, value):
    # key: None
    # value: 2d numpy array of shape (num_samples, num_dims) 
     
    num_data_points = value.shape[0]

    # idea: create a corset from the input data
    # step 1: D^2 sampling
    
    B_indices = np.zeros(k, dtype=int)
    B = np.zeros((k, num_dims), dtype=float)
    
    #choose first cluster center uniformly at random
    B_indices[0] = np.random.randint(500, size=1)[0]
    B[0] = value[B_indices[0]]
    
    probabilities = np.zeros(num_data_points)
 
    # computing the denominator for D^2 sampling     
    denominator = 0.0
    for index in range(0, num_data_points):
        denominator += compute_distance_to_closest_center(value[index], B[0])
      
    # computing the probabilities array
    for index in range(0, num_data_points):
        current_d = compute_distance_to_closest_center(value[index], B)
        probabilities[index] = current_d / (denominator - current_d)
        
    print(probabilities)           

#   for index in range(0, num_data_points):
#       print('inside for')


    #TODO change back to yield
    return 0, "value"  # all mappers should yield the same key


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(200, 250)


if __name__ == "__main__":

    # testing the compute_distance_to_closest_center function
#    centers_vector = np.array([[1.,1.],[2.,2.]])
#    data_point = np.array([4.,4.])
#    distance_to_closest_center = compute_distance_to_closest_center(data_point, centers_vector)
#    print(distance_to_closest_center)

    # testing the D^2 sampling
    value = np.random.rand(500,num_dims)
    mapper(None, value)





