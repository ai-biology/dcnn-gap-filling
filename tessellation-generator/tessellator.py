#!/usr/bin/env python3

"""
Create pairs of noisy and segmented images using voronoi tesselation
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from typing import Dict, List, Optional, Tuple
import warnings

import click
import numpy as np
from scipy.spatial import Voronoi
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import LineString, MultiLineString, box
from shapely.affinity import translate
from tqdm import tqdm


@dataclass
class Params:
    """ Definition of parameters for tesselation """

    width: int
    n_cells: int

    n_gaps: int
    gap_length_mu: float
    gap_length_sd: float

    line_strength_mu: float
    line_strength_sd: float

    line_width: int

    noise_sd: float
    blur_radius: float

    @classmethod
    def from_json_file(cls, filename):
        """ Load params from json file """
        with open(filename) as file:
            return cls(**json.load(file))


@dataclass
class Line:
    """ A single Line/Edge """

    id_: int
    coords: List[Tuple[float, float]]
    strength: int
    width: int
    gaps: List["Gap"]

    @classmethod
    def from_line_string(cls, id_, line, strength, width, gaps):
        """ Init from shapely.geometry.LineString """
        coords = list(map(tuple, np.array(line.coords).astype(float)))
        return cls(id_=id_, coords=coords, strength=strength, width=width, gaps=gaps)

    def interpolate(self, position) -> Tuple[float, float]:
        """
        Use shapely to interpolate relative position
        :returns: Coords of position along line
        """
        point = LineString(self.coords).interpolate(position, normalized=True)
        point = np.array(point).astype(float)
        return (point[0], point[1])

    def accumulate_gap_cover(self) -> float:
        """ Return percentage of missing length """
        return min(sum(gap.length for gap in self.gaps), 1.0)

    def as_dict(self):
        """ Convert to dict for summary table """
        return {
            "id": self.id_,
            "x1": self.coords[0][0],
            "y1": self.coords[0][1],
            "x2": self.coords[1][0],
            "y2": self.coords[1][1],
            "strength": self.strength,
            "width": self.width,
            "gap_cover": self.accumulate_gap_cover(),
            "n_gaps": len(self.gaps),
        }


@dataclass
class Gap:
    """ A single Gap """

    id_: int
    position: float  # relative position across line
    length: float  # relative length across line

    def coords_on_line(
        self, line: Line
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """ Get interpolated coordinates on line """
        start = line.interpolate(self.position)
        end = line.interpolate(self.position + self.length)
        return (start, end)

    def as_dict(self, line: Optional[Line] = None):
        """ Convert to dict, optionally computing props wrt to line """
        self_dict = asdict(self)
        self_dict["id"] = self_dict.pop("id_")

        if line is not None:
            coords = self.coords_on_line(line)
            self_dict.update(
                {
                    "x1": coords[0][0],
                    "y1": coords[0][1],
                    "x2": coords[1][0],
                    "y2": coords[1][1],
                    "width": line.width,
                }
            )

        return self_dict

    def manh_length(self, line: Line) -> float:
        """ Calculate manhattan length of gap, given its line """
        coords = self.coords_on_line(line)
        return abs(coords[0][0] - coords[1][0]) + abs(coords[0][1] - coords[1][1])


def sigmoid(x):
    return 1 / (1 + 100 ** (-x + 0.5))


def draw_voronoi_lines(img_width: int, n_cells: int) -> List[LineString]:
    """ Draw voronoi lines in a square image """
    centers = np.random.uniform(img_width, size=(n_cells, 2))
    vor = Voronoi(centers)
    lines = MultiLineString(
        [
            LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line
        ]
    )
    return lines


def crop_voronoi(lines: MultiLineString, img_width) -> MultiLineString:
    """ Ensure all lines are within img boundaries """
    bounds = box(0, 0, img_width, img_width)
    return lines.intersection(bounds)


def random_gaps(length_mu, length_sd, size, n_lines) -> Dict[int, List[Gap]]:
    """
    Draw gaps distributed uniformly on the lines
    as (position, length, line_index) tuples
    """
    # draw relative lengths and positions in (0, 1)
    gap_lengths = np.random.normal(length_mu, length_sd, size)
    # gap_lengths = np.clip(gap_lengths, 0, 1)
    gap_lengths = sigmoid(gap_lengths)

    # draw positions from (0, 1-length)
    gap_positions = np.random.uniform(0, 1 - gap_lengths, size=size)

    gaps = [
        Gap(id_=i, position=pos, length=length)
        for i, (pos, length) in enumerate(zip(gap_positions, gap_lengths))
    ]

    # draw lines the gaps will be applied to
    if size > n_lines:
        warnings.warn("Less lines than requested gaps, increase n_cells")
        size = n_lines

    line_idxs = np.random.choice(n_lines, size=size, replace=False)

    gap_dict: Dict[int, List[Gap]] = defaultdict(list)
    for idx, gap in zip(line_idxs, gaps):
        gap_dict[idx].append(gap)

    return gap_dict


def reject_gaps_by_length(
    line: Line,
    gaps: List[Gap],
    min_manh_length: Optional[float],
    max_manh_length: Optional[float],
) -> List[Gap]:
    """ Remove gaps that are shorter than min_manh_length """
    gaps = iter(gaps)

    if min_manh_length is not None:
        gaps = (gap for gap in gaps if gap.manh_length(line) > min_manh_length)

    if max_manh_length is not None:
        gaps = (gap for gap in gaps if gap.manh_length(line) < max_manh_length)

    return list(gaps)


def random_line_strengths(line_strength_mu, line_strength_sd, size):
    """ Generate line strengths as uint8 """
    line_strengths = np.random.normal(line_strength_mu, line_strength_sd, size)
    line_strengths = np.clip(line_strengths, 0, 1) * 256
    line_strengths = line_strengths.astype(int).tolist()
    return line_strengths


def add_noise(img, noise_sd, blur_radius):
    """ Add gaussian noise and apply gaussian blur """
    noise = np.random.normal(0, noise_sd, img.size) * 255

    noisy = np.array(img).astype(int) + noise.astype(int)
    noisy = np.clip(noisy, 0, 255).astype("uint8")
    img = Image.fromarray(noisy, "L")

    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


def get_line_table(lines: List[Line]) -> pd.DataFrame:
    """ Return DataFrame with line descriptions """
    lines_df = pd.DataFrame.from_records([line.as_dict() for line in lines], index="id")
    lines_df.index.name = "line_id"
    return lines_df


def get_gap_table(lines: List[Line]) -> pd.DataFrame:
    """ Return DataFrame with gap descriptions """
    records = {}
    for line in lines:
        for gap in line.gaps:
            records[(line.id_, gap.id_)] = gap.as_dict(line)

    gaps_df = pd.DataFrame.from_dict(records, "index")

    if not gaps_df.empty:
        gaps_df.index.names = ["line_id", "gap_id"]

    return gaps_df


def render_gap(img_width, line: Line, gap: Gap) -> Image:
    """ Render a single gap """
    img = Image.new("L", (img_width, img_width), color=0)
    draw = ImageDraw.Draw(img)

    gap_start, gap_end = gap.coords_on_line(line)
    width = line.width + 2
    draw.line([*gap_start, *gap_end], width=width, fill=255)

    return img


def render_observation(img_width, lines: List[Line]) -> Image:
    """ Render a noisy observation """
    img = Image.new("L", (img_width, img_width), color=255)
    draw = ImageDraw.Draw(img)

    # draw all lines
    for line in lines:
        draw.line(line.coords, width=line.width, fill=line.strength)

    # put gaps in place
    for line in lines:
        for gap in line.gaps:
            # set gap line width a bit wider than line to delete line cleaner
            width = line.width + 2
            gap_start, gap_end = gap.coords_on_line(line)
            draw.line([*gap_start, *gap_end], width=width, fill=255)

    return img


def render_segmentation(img_width, lines: List[Line]) -> Image:
    """ Render a segmented ground truth """
    img = Image.new("L", (img_width, img_width), color=255)
    draw = ImageDraw.Draw(img)

    for line in lines:
        draw.line(line.coords, width=line.width, fill=0)

    return img


def render_gaps_diff(segmented: Image, observed: Image) -> Image:
    """ Returns the difference of the images """
    observed_signal = (np.array(observed) != 255) * 255
    segmented_inv = np.array(segmented)
    gaps = segmented_inv + observed_signal
    return Image.fromarray(gaps.astype("uint8"))


def render_gaps_single(img_width, segmented, lines: List[Line]):
    """ Render each gap in a single image """
    # convert segmentation to mask to remove overflowing gap pixels
    segmented = ~np.array(segmented, dtype=bool)

    images = []
    for line in lines:
        for gap in line.gaps:
            img = render_gap(img_width, line, gap)
            img = np.array(img, dtype=bool)

            # intersection with segmented to remove overflowing gap pixels
            img = segmented & img

            img = Image.fromarray(img.astype("uint8") * 255)

            img.line_id = line.id_
            img.gap_id = gap.id_
            images.append(img)
    return images


def tessellate(params):
    """ Generate tessellated image """
    lines = draw_voronoi_lines(params.width, params.n_cells)
    lines = crop_voronoi(lines, params.width)

    line_strengths = random_line_strengths(
        params.line_strength_mu, params.line_strength_sd, len(lines)
    )

    gaps = random_gaps(
        params.gap_length_mu, params.gap_length_sd, params.n_gaps, len(lines)
    )

    lines = [
        Line.from_line_string(
            id_=i, line=line, strength=strength, width=params.line_width, gaps=gaps[i]
        )
        for i, (line, strength) in enumerate(zip(lines, line_strengths))
    ]

    line_table = get_line_table(lines)
    gap_table = get_gap_table(lines)

    segmented = render_segmentation(params.width, lines)
    observed = render_observation(params.width, lines)
    gaps_diff = render_gaps_diff(segmented, observed)
    gaps_single = render_gaps_single(params.width, segmented, lines)

    observed = add_noise(observed, params.noise_sd, params.blur_radius)

    return observed, segmented, gaps_diff, gaps_single, line_table, gap_table


@click.command()
@click.option(
    "-p", "--params", type=click.Path(readable=True, exists=True), required=True
)
@click.option("-n", "--n-images", type=int, default=1)
@click.option(
    "-o", "--outpath", type=click.Path(exists=True, allow_dash=True), default="-"
)
@click.option("-s", "--seed", default=42)
def main(params, n_images, outpath, seed):
    """ Generate tessellations """
    np.random.seed(seed)

    params = Params.from_json_file(params)

    line_tables = []
    gap_tables = []

    # create output folder for gaps
    if not os.path.exists(f"{outpath}/gaps"):
        os.mkdir(f"{outpath}/gaps")

    for i in tqdm(range(n_images)):
        observed, segmented, gaps_diff, gaps_single, line_table, gap_table = tessellate(
            params
        )

        line_tables.append(line_table)
        gap_tables.append(gap_table)

        if outpath != "-":
            segmented.save(f"{outpath}/segmented_{i}.png")
            observed.save(f"{outpath}/observed_{i}.png")
            gaps_diff.save(f"{outpath}/gaps_{i}.png")

            for gap_img in gaps_single:
                name = f"gap_{i}_{gap_img.line_id}_{gap_img.gap_id}.png"
                gap_img.save(f"{outpath}/gaps/{name}")

        else:
            segmented.show()
            observed.show()

    line_tables = pd.concat(line_tables, keys=range(n_images))
    gap_tables = pd.concat(gap_tables, keys=range(n_images))

    if outpath != "-":
        line_tables.to_csv(f"{outpath}/lines.csv")
        gap_tables.to_csv(f"{outpath}/gaps.csv")


if __name__ == "__main__":
    main()
