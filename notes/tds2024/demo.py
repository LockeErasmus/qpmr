"""

"""
from qpmr import qpmr
from manim import *



class DemoScene(Scene):


    def prepare_qpmr(self):
        region = [-2, 7, 0, 9]
        delays = np.array([0.0, 1.3, 3.5, 4.3])
        coefs = np.array([[20.1, 0, 0.2, 1.5],
                        [0, -2.1, 0, 1],
                        [0, 3.2, 0, 0],
                        [1.4, 0, 0, 0]])
        roots, meta = qpmr(region, coefs, delays)
        self.qpmr_roots = roots
        self.qpmr_meta = meta

    def construct_initial_scene(self):
        title = Tex(r"QPmR v2 algorithm")
        self.play(
            FadeIn(title)
        )
        # TODO add logo
        self.wait()
        self.remove(title)
        self.wait()

    def construct_introduction(self):
        title = Tex(r"Consider the quasipolynomial")
        eq_quasipoly = MathTex(
            r'h(s) = \left(1.5s^3 + 0.2s^2 + 20.1\right) + \left(s^3 -2.1 s\right)e^{-1.3s} + 3.2 s e^{-3.5s} + 1.4e^{-4.3s}'
        ).set(font_size=20)
        VGroup(title, eq_quasipoly).arrange(DOWN)
        
        self.play(
            Write(title),
            FadeIn(eq_quasipoly, shift=DOWN),
        )
        self.wait()
        self.remove(title, eq_quasipoly)

    
    def construct_cplane(self):

        plane = ComplexPlane(
            x_range = (-5., 10., 1),
            y_range = (-3., 12., 1),
            x_length = 6,
            y_length = 6,
        ).move_to(RIGHT*3).add_coordinates()

        self.play(
            Create(plane)
        )
        self.wait()
        # rect = Polygon()

        # construct region of interest
        region = [-2, 7, 0, 9]
        region_coords = [
            region[0] + 1j*region[2],
            region[1] + 1j*region[2],
            region[1] + 1j*region[3],
            region[0] + 1j*region[3],
        ]
        
        dots = [Dot(plane.n2p(rc), color=RED)  for rc in region_coords]
        rect = Polygon(*[plane.number_to_point(rc) for rc in region_coords], color=RED)
        # texts = [
        #     MathTex(f'{region[0] + 1j*region[2]}').next_to(dots[0], DOWN),
        # ]
        self.add(
            *dots,
            rect,
        )
        rect2 = rect.save_state()
        self.play(
            rect.animate.set_fill(RED, opacity=0.3)
        )
        self.play(
            rect.animate.restore()
        )

        real_paths = []
        for contour_re in self.qpmr_meta.contours_real:
            contour_points = contour_re[:,0] + contour_re[:, 1] * 1j
            path = VMobject(fill_color=BLUE,)
            path.set_points_smoothly([plane.n2p(n) for n in contour_points])
            real_paths.append(path)

        imag_paths = []
        for contour_im in self.qpmr_meta.contours_imag:
            contour_points = contour_im[:,0] + contour_im[:, 1] * 1j
            path = VMobject(fill_color=GREEN)
            path.set_points_smoothly([plane.n2p(n) for n in contour_points])
            imag_paths.append(path)
        
        self.add(
            *real_paths
        )
        self.wait()
        self.add(
            *imag_paths
        )

        self.wait()

    def construct(self):
        # calculates QPmR results
        self.prepare_qpmr()
        
        #self.construct_initial_scene()
        self.construct_cplane()

        
        
        
