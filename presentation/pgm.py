from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

pgm = daft.PGM([4.7, 2.35], origin=[-1.35, 2.2], observed_style="inner")

pgm.add_node(daft.Node("Omega", r"", -1, 4, fixed="true"))
pgm.add_node(daft.Node("theta", r"$\theta$", 0, 4))
pgm.add_node(daft.Node("Mc", r"$M_n,c_n$", 1, 4, aspect=1.5))
pgm.add_node(daft.Node("2MASS", r"\textsl{\small{2MASS}}", 3, 4.2, fixed=True))
pgm.add_node(daft.Node("dust", r"dust", 3, 3.6, fixed=True))
pgm.add_node(daft.Node("Gaia", r"\textsl{Gaia}", 3, 3, fixed=True))
pgm.add_node(daft.Node("JK", r"$J_n,K_n$", 2, 4, observed=True, aspect=1.5))
pgm.add_node(daft.Node("D", r"$D_n$", 1, 3))
pgm.add_node(daft.Node("MW", r"MW", 0, 3, fixed=True))
pgm.add_node(daft.Node("parallax", r"$\varpi_n$", 2, 3, observed=True))

pgm.add_plate(daft.Plate([0.5, 2.25, 2, 2.25],
        label=r"stars $n$"))

pgm.add_edge("Omega", "theta")
pgm.add_edge("theta", "Mc")
pgm.add_edge("Mc", "JK")
pgm.add_edge("2MASS", "JK")
pgm.add_edge("dust", "JK")
pgm.add_edge("MW", "D")
pgm.add_edge("D", "parallax")
pgm.add_edge("D", "JK")
pgm.add_edge("Gaia", "parallax")

pgm.render()
pgm.figure.savefig("Anderson.pdf")
pgm.figure.savefig("Anderson.png", dpi=200)
