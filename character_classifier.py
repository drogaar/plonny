import plonny

l0 = plonny.Input((28, 28, 1))

l1 = plonny.Conv2D(l0, (28, 28, 32), (5,5))
l2 = plonny.Pool(l1, (14, 14, 32))

l3 = plonny.Conv2D(l2, (14, 14, 64), (8,8))
l4 = plonny.Pool(l3, (7, 7, 64))

l5 = plonny.FullyConnected(l4, 1024)

l5.graphshow("Character classifier")
