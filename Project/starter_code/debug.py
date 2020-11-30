from matplotlib import pyplot as plt
from DataLoader import *
from ImageUtils import *
# data_dir = "./Project/cifar-10-batches-py/"
data_dir = "../cifar-10-batches-py/"
x_train, y_train, x_test, y_test = load_data(data_dir)
img = parse_record(x_train[0],False)
print(img.shape)
plt.imshow(img/255)

im = img
plt.imshow(im/255)
# ix = (0,0)
# print(subsample_ix(ix,2))
# print()
# print(interp2d(ix,im))
# plt.imshow(im)

# interp2d((0,-1),im)

plt.subplot(5,10,1)
for i in range(1,51):
    plt.subplot(5,10,i)
    r_im = parse_record(x_train[np.random.randint(x_train.shape[0])],True)
    # r_im = random_transform(im)
    plt.imshow((r_im + 1)/2 )
plt.show()
# im = np.random.rand(5,5)
# print(im)
# ix = [1.5,2.21]
# interp2d(ix,im)