import plonny
import graph
import layergrid

if __name__ == '__main__':
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
    lg.set(1, 0, l0)

    lg.set(0, 1, l1)
    lg.set(0, 2, l2)
    lg.set(0, 3, l3)
    lg.set(0, 4, l4)
    lg.set(1, 5, l5)

    lg.set(2, 2, la1)
    lg.set(2, 3, la2)

    lg.set(0, 6, l6)
    print(lg.grid)

    graph.Graph(lg).graphshow("")
