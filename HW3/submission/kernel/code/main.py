from DataReader import prepare_data
from model import Model

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def main():
    # ------------Data Preprocessing------------
    train_X, train_y, valid_X, valid_y, train_valid_X, train_valid_y, test_X, test_y = prepare_data(data_dir, train_filename, test_filename)

    # ------------Kernel Logistic Regression Case------------
    ### YOUR CODE HERE
    learning_rate = 0.001
    max_epoch = 30
    batch_size = train_valid_X.shape[0]
    # sigmas = [1,5,10,15,18,20,40]
    # scores = []
    # for sigma in sigmas:
    #     model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    #     model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(test_X, test_y))
    # for i in range(len(sigmas)):
    #     print("{:d}&{:.2f}\\\\".format(sigmas[i], scores[i]*100))
    
    sigma = 19
    max_epoch = 100
    model = Model('Kernel_LR', train_valid_X.shape[0], sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------RBF Network Case------------
    ### YOUR CODE HERE
    learning_rate = 0.001
    max_epoch = 30
    batch_size = 64
    sigma = 19
    # hds = [1,2,4,8,12,16,32]
    # scores = []
    # for hd in hds:
    #     model = Model('RBF', hd, sigma)
    #     model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(valid_X, valid_y))
    # for i in range(len(hds)):
    #     print("{:d}&{:.2f}\\\\".format(hds[i], scores[i]*100))
    
    hidden_dim = 12
    model = Model('RBF', hidden_dim, sigma)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE

    # ------------Feed-Forward Network Case------------
    ### YOUR CODE HERE
    # Run your feed-forward network model here
    learning_rate = 0.001
    max_epoch = 30
    batch_size = 64
    # hds = [1,2,4,8,12,16,32]
    # scores = []
    # for hd in hds:
    #     model = Model('FFN', hd)
    #     model.train(train_X, train_y, valid_X, valid_y, max_epoch, learning_rate, batch_size)
    #     scores.append(model.score(test_X, test_y))
    # for i in range(len(hds)):
    #     print("{:d}&{:.2f}\\\\".format(hds[i], scores[i]*100))
    
    hidden_dim = 12
    model = Model('FFN', hidden_dim)
    model.train(train_valid_X, train_valid_y, None, None, max_epoch, learning_rate, batch_size)
    score = model.score(test_X, test_y)
    print("score = {} in test set.\n".format(score))
    ### END YOUR CODE
    
if __name__ == '__main__':
    main()