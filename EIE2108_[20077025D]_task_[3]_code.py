import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

in_image_filename = '/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/myTimg.png'
encoding_result_filename = '/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/encoded.out'
reconstructed_image_filename = "/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/reconstructed.png"
shape = mpimg.imread(in_image_filename).shape
R = shape[0]
C = shape[1]
x = y = 0
d = 4


# Menu
def menu():
    print('''=== Menu ===
[f] select a file to be compressed
[d] set parameter d
[c] check current information
[s] start compress
[q] quit''')
    choice = input('Enter your choice (f,d,c,s,q): ')
    if choice == 'f':
        input_file_name()
    elif choice == 'd':
        block_size_d()
        print('Successfully set d to', d, '\n')
    elif choice == 'c':
        check_information()
    elif choice == 's':
        BVQCencode(in_image_filename, encoding_result_filename, d)
        BVQCdecode(encoding_result_filename, reconstructed_image_filename)
        evaluate(origin_image, reconstructed_image)
    elif choice == 'q':
        print('Thanks and bye~')
        return 0
    else:
        print("Invalid input. Please input again!\n")
    return 1


# input the file name
def input_file_name():
    global in_image_filename, shape, R, C
    # in_image_filename="/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/myTimg.png" '/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/reconstructed.png'
    in_image_filename = input('Please input the file path of an image file to Compress: ')
    if in_image_filename == '':
        print('Default file "/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/myTimg.png" is being used.\n')

        in_image_filename = "/Users/caizhuoang/Downloads/FIMT-yr21-task3-dataset/myTimg.png"

    elif not os.path.exists(in_image_filename):
        print('Path is not found. Please enter another one.')
        input_file_name()
    # Help the user to input png format files.
    elif in_image_filename[-4:] != '.png':
        print('Assigned file is not in proper file format. Please input a file in png format.')
        input_file_name()
    else:
        print('File have been successfully read!\n')
        shape = mpimg.imread(in_image_filename).shape
        R = shape[0]
        C = shape[1]


# block size d
def block_size_d():
    global d, D
    dloop = 1
    while dloop:
        d = input('Please input a valid block size(must be an integer power of 2): ')
        if not d.isdigit():
            print('Input must be integer.')
            block_size_d()
        else:  # Judge whether the input is power of 2.
            D = int(d)
            while D > 1:
                D /= 2
            if D != 1:
                print('Input is not a power of 2.')
                continue
            dloop = 0
    d = int(d)


def check_information():
    print('selected file: {}\nd: {}\nR: {}\nC: {}\nx: {}\ny: {}\n'.format(in_image_filename, d, R, C, x, y))


# decode index of subblock from decimal to plane
def decode_sub_index(x):
    a = bin(x)[2:].zfill(8)
    index_plane = [int(a[0:2], 2), int(a[2:4], 2), int(a[4:6], 2), int(a[6:], 2)]
    return index_plane


def evaluate(A, B):
    global R, C
    mse = 0
    for i in range(0, R):
        for j in range(0, C):
            mse = mse + (A[i][j] - B[i][j]) ** 2
    mse = mse / (R * C)
    PPSNR = 10 * math.log10(255 * 255 / mse)
    print('peak-to-peak singal-to-noise ratio is {}'.format(PPSNR), '\n')


def BVQCencode(in_image_filename, out_encoding_result_filename, d):
    global x, y, R, C, origin_image
    img = mpimg.imread(in_image_filename)  # store the image into img
    X = img * 255  # Normalize its intensity values such that each pixel value is bounded in [0,255].
    X = X[x:x + R, y:y + C]
    origin_image = X  # To evaluate
    print('\nOriginal image:')
    plt.imshow(X, cmap='gray')
    plt.show()
    print('shape', img.shape)

    # Partition the image into blocks. Each block is of size d*d pixels.
    block_num_of_col = int(X.shape[0] / d)
    block_num_of_row = int(X.shape[1] / d)
    col3 = block_num_of_col % 256
    col4 = block_num_of_col // 256
    row5 = block_num_of_row % 256
    row6 = block_num_of_row // 256

    file = open(out_encoding_result_filename, 'wb')
    # Header
    header = np.array([6, d, col3, col4, row5, row6], dtype='uint8')
    for byte in header:
        file.write(byte)
    for I in range(0, X.shape[0], d):
        for J in range(0, X.shape[1], d):
            x = X[I:I + d, J:J + d]
            u = np.mean(x)  # Compute the mean of each block
            std = np.std(x)  # Compute the standard deviation of each block
            stored_data = [round(u), round(std)]
            for byte in stored_data:
                file.write(np.uint8(byte))
            # Contrast a codebook
            g0 = max(0, u - std)
            g1 = min(255, u + std)
            c0 = np.array([[g0, g0], [g1, g1]])
            c1 = np.array([[g1, g1], [g0, g0]])
            c2 = np.array([[g0, g1], [g0, g1]])
            c3 = np.array([[g1, g0], [g1, g0]])
            c = [c0, c1, c2, c3]
            sub_index_plane = []

            '''Divide the block into subblocks of size 2x2 and approximate each of them 
            as the closest codeword based on their distances from the subblock.'''
            for i in range(0, d, 2):
                for j in range(0, d, 2):
                    dis = []
                    sub = x[i:i + 2, j:j + 2]

                    for k in range(4):
                        dis.append(((sub - c[k]) * (sub - c[k])).sum())

                    sub_index_plane.append(dis.index(min(dis)))
                    if len(sub_index_plane) == 4:
                        Isb = np.dot(sub_index_plane, [64, 16, 4, 1])
                        file.write(np.uint8(Isb))
                        sub_index_plane = []
            if sub_index_plane != []:
                sub_index_plane.extend([0] * (4 - len(sub_index_plane)))
                Isb = np.dot(sub_index_plane, [64, 16, 4, 1])
                file.write(np.uint8(Isb))
                sub_index_plane = []
    file.close()
    x = y = 0
    R = shape[0]
    C = shape[1]


def BVQCdecode(in_encoding_result_filename, out_reconstructed_image_filename):
    file = open(in_encoding_result_filename, 'rb')

    # read the header
    header_len = file.read(1)[0]
    d = file.read(1)[0]
    no_of_block_rows = file.read(1)[0] + file.read(1)[0] * 256
    no_of_block_cols = file.read(1)[0] + file.read(1)[0] * 256

    file.read(header_len - 6)

    OImg = np.zeros([no_of_block_rows * d, no_of_block_cols * d])

    read_bytes = int((d / 4) ** 2 + 2 + 0.999)

    for i in range(no_of_block_rows):
        for j in range(no_of_block_cols):

            stored_data = file.read(read_bytes)
            u = stored_data[0]
            std = stored_data[1]
            Isb = stored_data[2:]

            plane = []

            for isb in Isb:
                plane.extend(decode_sub_index(isb))

            g0 = max(0, u - std)
            g1 = min(255, u + std)
            c0 = np.array([[g0, g0], [g1, g1]])
            c1 = np.array([[g1, g1], [g0, g0]])
            c2 = np.array([[g0, g1], [g0, g1]])
            c3 = np.array([[g1, g0], [g1, g0]])
            c = [c0, c1, c2, c3]
            # print(g0,g1,plane)

            block = np.zeros([d, d])
            m = 0
            for k in range(0, d, 2):
                for l in range(0, d, 2):
                    # print(k,l,m,plane)
                    block[k:k + 2, l:l + 2] = c[plane[m]]
                    m += 1

            OImg[i * d:i * d + d, j * d:j * d + d] = block

    file.close()
    global reconstructed_image
    reconstructed_image = OImg
    plt.imshow(OImg, cmap='gray', vmax=255, vmin=0)
    plt.imsave(reconstructed_image_filename, OImg, cmap='gray', vmax=255, vmin=0)
    print('\nProcessed image:')
    plt.show()
    print('shape', OImg.shape)
    # print('\n')


loop = 1
while loop:
    loop = menu()
