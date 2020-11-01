import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)




# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 3
hidden_dim = 24
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

        
running = True
while running:
    cmd = input("Enter (T)est, T(r)ain, (E)xit ").strip().upper()

    if cmd == "T":
            # create a simple addition problem (a + b = c)
            num = False
            while not num:
                try:
                    a_int = int(input("Enter first number between 1 and {}:\n".format(str(largest_number/3)))) # int version
                except:
                    print("Incorrect argument. Please enter a interger.\n")
                if not (a_int > (largest_number/3)):
                    num = True
                else:
                    print("number must be between 1 and {}".format(str(largest_number/3)))
            a = int2binary[a_int] # binary encoding
            num = False
            while not num:
                try:
                    b_int = int(input("Enter next number between 1 and {}:\n".format(str(largest_number/3)))) # int version
                except:
                    print("Incorrect argument. Please enter a interger.\n")
                if not (b_int > (largest_number/3)):
                    num = True
                else:
                    print("number must be between 1 and {}\n".format(str(largest_number/3)))
            b = int2binary[b_int] # binary encoding
            num = False
            while not num:
                try:
                    c_int = int(input("Enter last number between 1 and {}:\n".format(str(largest_number/3)))) # int version
                except:
                    print("Incorrect argument. Please enter a interger.\n")
                if not (c_int > (largest_number/3)):
                    num = True
                else:
                    print("number must be between 1 and {}\n".format(str(largest_number/3)))
            c = int2binary[c_int] # binary encoding

            print("\n------------\n")

            # true answer
            d_int = a_int + b_int + c_int
            d = int2binary[d_int]
    
            # where we'll store our best guess (binary encoded)
            e = np.zeros_like(d)

            overallError = 0
    
            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(hidden_dim))
    
            for position in range(binary_dim):
                # generate input and output
                X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1],c[binary_dim - position - 1]]])
                y = np.array([[d[binary_dim - position - 1]]]).T

                # hidden layer (input ~+ prev_hidden)
                layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

                # output layer (new binary representation)
                layer_2 = sigmoid(np.dot(layer_1,synapse_1))

                # did we miss?... if so, by how much?
                layer_2_error = y - layer_2
                layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
                overallError += np.abs(layer_2_error[0])
    
                # decode estimate so we can print it out
                e[binary_dim - position - 1] = np.round(layer_2[0][0])
        
                # store hidden layer so we can use it in the next timestep
                layer_1_values.append(copy.deepcopy(layer_1))

            print("Error:" + str(overallError))
            print("Pred:" + str(e))
            print("True:" + str(d))
            out = 0
            for index,x in enumerate(reversed(e)):
                out += x*pow(2,index)
            print(str(a_int) + " + " + str(b_int) + " + " + str(c_int) + " = " + str(out))
            print("------------")

    elif cmd == "R":
        num = False
        while not num:
            try:
                arg1 = int(input("Enter number of training itterations:\n"))
                num = True
            except:
                print("Incorrect argument. Please enter a interger.")
        num = False
        while not num:
            try:
                arg2 = int(input("Display progress every ___ itterations:\n"))
                num = True
            except:
                print("Incorrect argument. Please enter a interger.")

        print("\n------------\n")

        for j in range(arg1):
            # generate a simple addition problem (a + b = c)
            a_int = np.random.randint(largest_number/3) # int version
            a = int2binary[a_int] # binary encoding

            b_int = np.random.randint(largest_number/3) # int version
            b = int2binary[b_int] # binary encoding

            c_int = np.random.randint(largest_number/3) # int version
            c = int2binary[c_int] # binary encoding


            # true answer
            d_int = a_int + b_int + c_int
            d = int2binary[d_int]
    
            # where we'll store our best guess (binary encoded)
            e = np.zeros_like(d)

            overallError = 0
    
            layer_2_deltas = list()
            layer_1_values = list()
            layer_1_values.append(np.zeros(hidden_dim))
    
            # moving along the positions in the binary encoding
            for position in range(binary_dim):
        
                # generate input and output
                X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1],c[binary_dim - position - 1]]])
                y = np.array([[d[binary_dim - position - 1]]]).T

                # hidden layer (input ~+ prev_hidden)
                layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

                # output layer (new binary representation)
                layer_2 = sigmoid(np.dot(layer_1,synapse_1))

                # did we miss?... if so, by how much?
                layer_2_error = y - layer_2
                layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
                overallError += np.abs(layer_2_error[0])
    
                # decode estimate so we can print it out
                e[binary_dim - position - 1] = np.round(layer_2[0][0])
        
                # store hidden layer so we can use it in the next timestep
                layer_1_values.append(copy.deepcopy(layer_1))
    
            future_layer_1_delta = np.zeros(hidden_dim)
    
            for position in range(binary_dim):
        
                X = np.array([[a[position],b[position],c[position]]])
                layer_1 = layer_1_values[-position-1]
                prev_layer_1 = layer_1_values[-position-2]
        
                # error at output layer
                layer_2_delta = layer_2_deltas[-position-1]
                # error at hidden layer
                layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

                # let's update all our weights so we can try again
                synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                synapse_0_update += X.T.dot(layer_1_delta)
        
                future_layer_1_delta = layer_1_delta
    

            synapse_0 += synapse_0_update * alpha
            synapse_1 += synapse_1_update * alpha
            synapse_h += synapse_h_update * alpha    

            synapse_0_update *= 0
            synapse_1_update *= 0
            synapse_h_update *= 0
    
            # print out progress
            if(j % arg2 == 0):
                print("Error:" + str(overallError))
                print("Pred:" + str(e))
                print("True:" + str(d))
                out = 0
                for index,x in enumerate(reversed(e)):
                    out += x*pow(2,index)
                print(str(a_int) + " + " + str(b_int) + " + " + str(c_int) + " = " + str(out))
                print("------------")
    elif cmd == "E":
        print("Closing...")
        running = False
    else:
        print("Incorrect argument. Please enter one of the following: T, R, E")