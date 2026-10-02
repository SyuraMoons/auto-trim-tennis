# Adapted from AymenMB/Tennis-Match-Analysis- (court_reference_model.py): drawing and matplotlib removed.
# Reference court units are ~100 per metre (doubles width 1093 = 10.97 m, length 2374 = 23.77 m).

UNITS_PER_M = 100.0


class CourtReference:
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))

        self.key_points = [*self.baseline_top, *self.baseline_bottom,
                           *self.left_inner_line, *self.right_inner_line,
                           *self.top_inner_line, *self.bottom_inner_line,
                           *self.middle_line]

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}

        # (x, y) line segments for drawing overlays
        self.lines = [self.baseline_top, self.baseline_bottom, self.net, self.left_court_line,
                      self.right_court_line, self.left_inner_line, self.right_inner_line,
                      self.middle_line, self.top_inner_line, self.bottom_inner_line]

    def court_corners(self):
        """Doubles court outline, clockwise from top-left."""
        (x0, y0), (x1, _) = self.baseline_top
        y1 = self.baseline_bottom[0][1]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def play_area(self, behind_m, side_m):
        """Court outline expanded by behind_m past each baseline and side_m past each sideline."""
        (x0, y0), (x1, _) = self.baseline_top
        y1 = self.baseline_bottom[0][1]
        b, s = behind_m * UNITS_PER_M, side_m * UNITS_PER_M
        return [(x0 - s, y0 - b), (x1 + s, y0 - b), (x1 + s, y1 + b), (x0 - s, y1 + b)]
