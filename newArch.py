import plonny
import graph
import layergrid

def net1():
    l0 = plonny.Input((32, 32, 1))

    la1 = plonny.Conv2D([l0], (8, 8, 40), (3,3))
    la2  = plonny.Custom((8, 8, 40), "Text\nembeds", [la1])

    l1 = plonny.Conv2D([l0], (32, 32, 40), (3,3))
    l2 = plonny.Pool((16, 16, 40), [l1])
    l3 = plonny.Conv2D([l2], (16, 16, 40), (3,3))
    l4 = plonny.Pool((8, 8, 40), [l3])
    l5 = plonny.Conv2D([l4, la2], (8, 8, 40), (3,3))
    l6  = plonny.Upsample((32, 32, 40), [l5])


    lg = layergrid.LayerGrid()
    lg.set(0, 0, l0)
    lg.set(1, 0, l0)

    lg.set(0, 1, l1)
    lg.set(0, 2, l2)
    lg.set(0, 3, l3)
    lg.set(0, 4, l4)

    lg.set(0, 5, l5)
    lg.set(1, 5, l5)

    lg.set(1, 2, la1)
    lg.set(1, 3, la2)

    lg.set(0, 6, l6)
    lg.set(1, 6, l6)
    # print(lg.get_row(0))
    print(lg)

    graph.Graph(lg).graphshow()

def net2():
    l0 = plonny.Input((400, 600, 1))
    l1 = plonny.Conv2D([l0], (400, 600, 32), (3, 3))
    l2 = plonny.Pool((200, 300, 32), [l1])
    l3 = plonny.Conv2D([l2], (200, 300, 64), (3,3))
    l4 = plonny.Pool((100, 150, 64), [l3])

    l41  = plonny.Custom((100, 150, 1), "Feature\n  Map", [l4])
    l42  = plonny.Custom((100, 150, 1), "Feature\n  Map", [l4])
    l43  = plonny.Custom((100, 150, 1), "Feature\n  Map", [l4])
    l44  = plonny.Custom((100, 150, 1), "Feature\n  Map", [l4])

    l5 = plonny.FullyConnected([l41, l42, l43, l44], 64)
    l6  = plonny.Custom((1, 1), "LSTM", [l5])

    lg = layergrid.LayerGrid()
    lg.set(0, 0, l0)
    lg.set(0, 1, l1)

    lg.set(0, 2, l2)
    lg.set(0, 3, l3)
    lg.set(0, 4, l4)
    lg.set(0, 5, l41)

    lg.set(1, 5, l42)
    lg.set(2, 5, l43)

    lg.set(3, 5, l44)
    lg.set(0, 6, l5)

    lg.set(0, 7, l6)

    print(lg)
    graph.Graph(lg).graphshow()

def net3():
    l0 = plonny.Input((64,64,3), img="./tmp/rois/h15m34s15_class=BaseTech_p=1_used=False.jpg")
    l1 = plonny.Conv2D([l0], (64, 64, 64), (5, 5))
    l2 = plonny.Pool((32, 32, 64), [l1])
    l3 = plonny.Conv2D([l2], (32, 32, 128), (5, 5))
    l4 = plonny.Pool((16, 16, 64), [l3])
    # l5 = plonny.Reshape((1, 32768), [l3])
    l5 = plonny.Fluid((1, 92), "Reshape", 32768, [l4])#shape, name, actual_size=5, layers=None
    l6 = plonny.FullyConnected([l5], 2048)
    # l6 = plonny.Dropout((1, 2048), [l5])
    l7 = plonny.Dropout([l6], 2048)
    l8 = plonny.FullyConnected([l7], 5)

    lg = layergrid.LayerGrid()
    lg.set(0,0,l0)
    lg.set(0,1,l1)
    lg.set(0,2,l2)
    lg.set(0,3,l3)
    lg.set(0,4,l4)
    # lg.set(1,1,l0)
    lg.set(0,5,l5)
    lg.set(0,6,l6)
    lg.set(0,7,l7)
    # lg.set(0,6,l8)
    lg.set(0,8,l8)

    print(lg)
    graph.Graph(lg).graphshow("Box classifier")


if __name__ == '__main__':
    net1()
