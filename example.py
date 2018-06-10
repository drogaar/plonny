import plonny

if __name__ == "__main__":
    # Define model by calling layers
    image_0 = plonny.Input((32, 128, 1))

    conv2_1 = plonny.Conv2D(image_0, (32, 128, 16), (3,3))
    Pool_2  = plonny.Pool(conv2_1, (16, 64, 16))

    conv2_3 = plonny.Conv2D(Pool_2, (16, 64, 64), (3,3))
    Pool_4  = plonny.Pool(conv2_3, (8, 32, 64))

    conv2_5 = plonny.Conv2D(Pool_4, (8, 32, 64), (9,9))
    Pool_6  = plonny.Pool(conv2_5, (4, 16, 64))

    conv2_7 = plonny.Conv2D(Pool_6, (4, 16, 64), (3,3))
    conv2_8 = plonny.Conv2D(conv2_7, (4, 16, 16), (3,3))
    conv2_9 = plonny.Conv2D(conv2_8, (4, 16, 7), (3,3))

    resh_10 = plonny.Reshape(conv2_9, (8, 56))
    conv_11 = plonny.Conv2D(resh_10, (8, 28), (3,3))

    fccc_12 = plonny.FullyConnected(conv_11, 8)

    fccc_13 = plonny.FullyConnected(fccc_12, 50)

    fccc_13.graphshow("Neural Network")
