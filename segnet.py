import plonny

if __name__ == '__main__':
    l0 = plonny.Input((32, 32, 1))
    l1 = plonny.Conv2D(l0, (32, 32, 40), (3,3))
    l2 = plonny.Conv2D(l1, (32, 32, 40), (3,3))
    l3 = plonny.Pool(l2, (16, 16, 40))
    l4 = plonny.Conv2D(l3, (16, 16, 40), (7,7))
    l5 = plonny.Conv2D(l4, (16, 16, 40), (7,7))
    l6 = plonny.Pool(l5, (8, 8, 40))
    l7 = plonny.Upsample(l6, (16, 16, 40))
    l8 = plonny.Conv2D(l7, (16, 16, 40), (7,7))
    l9 = plonny.Conv2D(l8, (16, 16, 40), (7,7))
    l10 = plonny.Dropout(l9, (16, 16, 40))
    l11 = plonny.Upsample(l10, (32, 32, 40))
    l12 = plonny.Concat(l11, [l0])
    l13 = plonny.Conv2D(l12, (32, 32, 40), (3,3))
    l14 = plonny.Conv2D(l13, (32, 32, 26), (1,1))

    l14.graphshow("Segnet")
