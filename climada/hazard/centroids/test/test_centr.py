"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test CentroidsVector and CentroidsRaster classes.
"""
import unittest
from pathlib import Path

import numpy as np
import geopandas as gpd
import shapely

from climada import CONFIG
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import HAZ_TEMPLATE_XLS
from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coord
from climada.entity import Exposures
from rasterio import Affine

HAZ_TEST_MAT = Path(hazard_test.__file__).parent / "data" / "atl_prob_no_name.mat"


class TestCentroidsReader(unittest.TestCase):
    """Test read functions Centroids"""

    def test_centroid_pass(self):
        """Read a centroid excel file correctly."""
        centroids = Centroids.from_excel(HAZ_TEMPLATE_XLS)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 33.88)

    def test_base_grid(self):
        """Read new centroids using from_base_grid, then select by extent."""
        centroids = Centroids.from_base_grid(land=True, res_as=150)
        self.assertEqual(centroids.lat.size, 8858035)
        self.assertTrue(np.all(np.diff(centroids.lat) <= 0))

        count_sandwich = np.sum(centroids.region_id == 239)
        self.assertEqual(count_sandwich, 321)

        count_sgi = centroids.select(
            reg_id=239, extent=(-39, -34.7, -55.5, -53.6)  # south georgia island
        ).size
        self.assertEqual(count_sgi, 296)

        # test negative latitudinal orientation by testing that northern hemisphere (Russia)
        # is listed before southern hemisphere (South Africa)
        russia_max_idx = (centroids.region_id == 643).nonzero()[0].max()
        safrica_min_idx = (centroids.region_id == 710).nonzero()[0].min()
        self.assertTrue(russia_max_idx < safrica_min_idx)

    def test_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        gdf = gpd.GeoDataFrame(pd.read_excel(HAZ_TEMPLATE_XLS))
        gdf.geometry = gpd.points_from_xy(gdf["longitude"], gdf["latitude"])
        gdf["elevation"] = np.random.rand(gdf.geometry.size)
        gdf["region_id"] = np.zeros(gdf.geometry.size)
        gdf["region_id"][0] = np.NaN
        gdf["geom"] = gdf.geometry  # this should have no effect on centroids

        centroids = Centroids.from_geodataframe(gdf)

        for name, array in zip(["lat", "lon", "region_id"], [lat, lon, region_id]):
            np.testing.assert_array_equal(array, getattr(centroids, name))
        np.testing.assert_array_equal(extra, centroids.gdf.extra.values)
        self.assertTrue(u_coord.equal_crs(centroids.crs, crs))

    def test_from_geodataframe_invalid(self):
        # Creating an invalid GeoDataFrame with geometries that are not points
        invalid_geometry_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.Point((2, 2)),
                    shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                    shapely.LineString([(0, 1), (1, 0)]),
                ]
            }
        )

        with self.assertRaises(ValueError):
            # Trying to create Centroids from invalid GeoDataFrame
            Centroids.from_geodataframe(invalid_geometry_gdf)

    def test_from_exposures_with_region_id(self):
        """
        Test that the `from_exposures` method correctly extracts
        centroids and region_id from an `Exposure` object with region_id.
        """
        # Create an Exposure object with region_id, on_land and custom crs
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        value = np.array([1, 1, 1])
        region_id = np.array([1, 2, 3])
        on_land = [False, True, True]
        crs = "epsg:32632"
        gdf = gpd.GeoDataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "value": value,
                "region_id": region_id,
                "on_land": on_land,
            }
        )
        exposures = Exposures(gdf, crs=crs)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)
        self.assertEqual(centroids.crs, crs)

    def test_from_exposures_without_region_id(self):
        """
        Test that the `from_exposures` method correctly extracts
        centroids from an `Exposure` object without region_id.
        """
        # Create an Exposure object without region_id and variables to ignore
        # and default crs
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        value = np.array([1, 1, 1])
        impf_TC = np.array([1, 2, 3])
        centr_TC = np.array([1, 2, 3])
        gdf = gpd.GeoDataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "value": value,
                "impf_tc": impf_TC,
                "centr_TC": centr_TC,
            }
        )
        exposures = Exposures(gdf)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        self.assertEqual(centroids.lat.tolist(), lat.tolist())
        self.assertEqual(centroids.lon.tolist(), lon.tolist())
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))
        self.assertEqual(centroids.region_id, None)
        self.assertEqual(centroids.on_land, None)


class TestCentroidsWriter(unittest.TestCase):
    def test_write_hdf5(self):
        tmpfile = "test_write_hdf5.out.hdf5"
        xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon = 5, 6.5, -0.08, 0.12, 4, 5
        centr = Centroids.from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon)
        with self.assertLogs("climada.hazard.centroids.centr", level="INFO") as cm:
            centr.write_hdf5(tmpfile)
        self.assertEqual(1, len(cm.output))
        self.assertIn(f"Writing {tmpfile}", cm.output[0])
        centr.meta["nodata"] = None
        with self.assertLogs("climada.hazard.centroids.centr", level="INFO") as cm:
            centr.write_hdf5(tmpfile)
        self.assertEqual(2, len(cm.output))
        self.assertIn(
            "Skip writing Centroids.meta['nodata'] for it is None.", cm.output[1]
        )
        Path(tmpfile).unlink()


class TestCentroidsMethods(unittest.TestCase):
    def test_union(self):
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(latitude=lat, longitude=lon, on_land=on_land)

        lat2, lon2 = np.array([2, 3]), np.array([-2, 3])
        on_land2 = np.array([False, False])
        cent2 = Centroids(latitude=lat2, longitude=lon2, on_land=on_land2)

        lat3, lon3 = np.array([-1, -2]), np.array([1, 2])
        cent3 = Centroids(lat=lat3, lon=lon3)

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = cent1.union(cent1, cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = Centroids.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = cent1.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = Centroids.union(cent1, cent2, cent3)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3, -1, -2])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3, 1, 2])

    def test_union_meta(self):
        cent1 = Centroids.from_pnt_bounds((-1, -1, 0, 0), res=1)
        cent2 = Centroids.from_pnt_bounds((0, 0, 1, 1), res=1)
        cent3 = Centroids.from_lat_lon(np.array([1]), np.array([1]))

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0, 0, -1, -1, 1, 1, 0])
        np.testing.assert_array_equal(cent.lon, [-1, 0, -1, 0, 0, 1, 1])

        cent = cent3.union(cent1)
        np.testing.assert_array_equal(cent.lat, [1, 0, 0, -1, -1])
        np.testing.assert_array_equal(cent.lon, [1, -1, 0, -1, 0])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsData)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReader))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsMethods))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
