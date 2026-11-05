from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# WorldCover code -> class name
LANDCOVER_MAP = {
    50: "urban",
    40: "agricultural",
    10: "forest_semi_natural",   # tree cover
    20: "forest_semi_natural",   # shrubland
    30: "forest_semi_natural",   # grassland (ambiguous: pasture vs natural)
    60: "forest_semi_natural",   # bare / sparse vegetation
    70: "forest_semi_natural",   # snow and ice
    100: "forest_semi_natural",  # moss and lichen
    90: "wetlands",              # herbaceous wetland
    95: "wetlands",              # mangroves
    80: "water_bodies",
}

# class name -> integer label
CLASS_LABELS = {
    "urban": 1,
    "agricultural": 2,
    "forest_semi_natural": 3,
    "wetlands": 4,
    "water_bodies": 5,
}

class EODataSource(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


class OpenEOSource(EODataSource):
    def __init__(
        self,
        spatial_coordinates: tuple[float],
        temporal_extent: tuple[str],
        satellite: str = "SENTINEL2_L2A",
        bands: tuple[str] = ('B02', 'B03', 'B04', 'B08'), 
        max_cloud_cover: int = 50
    ):
        self.satellite = satellite
        self.spatial_coordinates = spatial_coordinates
        self.temporal_extent = temporal_extent
        self.bands = bands
        self.max_cloud_cover = max_cloud_cover
        self.connection = None

    def _establish_connection(self):
        import openeo
        self.connection = openeo.connect("openeo.dataspace.copernicus.eu")
        self.connection = self.connection.authenticate_oidc()

    def _load_collection(self):
        west, south, east, north = self.spatial_coordinates
        return self.connection.load_collection(
            self.satellite,
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=list(self.temporal_extent),
            bands=list(self.bands),
            max_cloud_cover=self.max_cloud_cover,
        )

    def _load_labels(self, reference_cube):
        west, south, east, north = self.spatial_coordinates
        labels = self.connection.load_collection(
            "ESA_WORLDCOVER_10M_2021_V2",
            spatial_extent={"west": west, "south": south, "east": east, "north": north},
            temporal_extent=["2021-01-01", "2022-01-01"],
            bands=["MAP"],
        )
        labels = labels.reduce_dimension(dimension="t", reducer="first")
        # "near" keeps class ids intact (no interpolation)
        return labels.resample_cube_spatial(reference_cube, method="near")


    # Other methods for masking clouds might be implemented
    @staticmethod
    def _mask_clouds(cube):
        return cube.process("mask_scl_dilation", data=cube, scl_band_name="SCL")

    # Other methods other than the median might be implemented
    @staticmethod
    def _temporal_median(cube):
        # (t, y, x, bands) -> (y, x, bands)
        return cube.reduce_dimension(dimension="t", reducer="median")

    # Other methods other than reflectance might be implemented 
    def _to_reflectance(self, cube):
        def scale(x):
            return x * 0.0001
        return cube.apply(scale)

    def _build_cube(self):
        cube = self._load_collection()
        #cube = self._mask_clouds(cube) # masking clouds is a very expensive step
        cube = self._temporal_median(cube)
        cube = self._to_reflectance(cube)
        return cube.merge_cubes(self._load_labels(cube))


    def _download(self):
        import rioxarray
        from rasterio.io import MemoryFile
        data = self._build_cube().save_result(format="GTiff").download()
        with MemoryFile(data) as mem, mem.open() as ds:
            return rioxarray.open_rasterio(ds, masked=True).load()

    def _to_dataframe(self, da) -> pd.DataFrame:
        long = (
            da.to_dataframe(name="value")
            .reset_index()
            .drop(columns="spatial_ref", errors="ignore")
        )
        cols = [*self.bands, "label"]
        names = {i: c.lower() for i, c in enumerate(cols, start=1)}
        df = (
            long.pivot(index=["y", "x"], columns="band", values="value")
                .rename(columns=names)
                .reset_index()
        )
        df.columns.name = None

        code = df["label"].round().astype("Int64")
        df["class"] = code.map(LANDCOVER_MAP)
        df["label"] = df["class"].map(CLASS_LABELS).astype("Int64")
        return df



    def load(self) -> pd.DataFrame:
        if self.connection is None:
            self._establish_connection()
        return self._to_dataframe(self._download())
