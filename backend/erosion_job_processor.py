import logging
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

import boto3
import geopandas as gpd
from constants import COLOURMAP_EROSION, DATA_DIR, OUTPUT_BUCKET
from eventbridge import put_event
from gis_utils.rendering.colourmap_helpers import create_basic_colourmap
from lidar_utils import (
    erosion_pipeline,
    fetch_inputs,
    laz_point_spacing_lastools,
    laz_tile,
    laz_tindex,
    reclassify_erosion,
)
from osgeo import gdal, osr
from s3 import s3_copy_file
from slugify import slugify

logger = logging.getLogger(__name__)


# TODO: need to adjust hexbin depending on point density?
# TODO: add noise filtering: filters.elm or filters.outlier
# TODO: figure out the relationship between tiling buffer and processing buffer
# TODO: LAStools on Linux doesn't support multicore processing --> use Python multiprocessing
# TODO: add tile-aware polygonization of the erosion heatmap
# TODO: estimate tile size as function of area size and GSD
# TODO: implement support for vertical EPSG
# TODO: speed up tiling through parallel processing of filters.crop (filters.splitter results in OOM)
# TODO: remove classify_ground() as assumes ground points are already classified?

# NOTE: the code assumes LiDAR data in LAZ format with projection information in the metadata
# NOTE: used Warp instead of Translate to avoid out-of-memory issues (slower though)
# NOTE: SciPy filters can't handle NaNs
# NOTE: SciPy filters runtime: uniform_filter < medfilt2d < median_filter
# NOTE: free LAStools: las2las, las2txt, lasindex, lasinfo, lasmerge, lasprecision, laszip, txt2las
# NOTE: buffers are not completely removed, but average PixelFunction slows down processing (x10 times)


def check_tile_target(bucket: str, prefix: str):
    """
    Ensure there is nothing in the target location for tiles.
    """
    s3 = boto3.client("s3")
    if s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=2)["KeyCount"] > 0:
        raise ValueError(f"There are already objects in the target tile prefix {prefix}")


def generate_erosion_heatmap(
    input: str,
    job_id: int,
    base_url: str,
    smooth_radius: int,
    tile_buffer: int,
    tile_size: int,
    z_threshold: float,
    gsd: float,
    cores: int,
):
    # TODO: don't copy locally, just use the files directly?
    fetch_inputs(input)
    os.chdir(DATA_DIR)
    nodata = -32768

    # Get number of physical cores (if not specified manually):
    if not cores:
        cores = multiprocessing.cpu_count() // 2

    # Verify average point spacing and projection in wkt:
    logger.info("Calculating average ground point spacing of reprocessed data and extracting WKT ...")
    GSD, PTS_COUNT, INPUT_WKT = laz_point_spacing_lastools(
        input_path=f"{DATA_DIR}", data_percent=1.0, input_ext=".laz", cores=cores
    )

    # Overwrite optimal GSD with user-defined GSD:
    if gsd != 0:
        logger.info(f"Overwriting optimal GSD {GSD} with {gsd}...")
        GSD = gsd

    logger.info(f"Output GSD: {GSD}")

    # Convert WKT to EPSG:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(INPUT_WKT)
    output_epsg = srs.GetAttrValue("AUTHORITY", 1)
    logger.info(f"Output EPSG: {output_epsg}")

    # Generate spatial index based (fast boundary):
    logger.info("Generating spatial index for original tiles ...")
    laz_tindex(f"{DATA_DIR}")

    files_to_process = glob(f"{DATA_DIR}/*.laz")
    gdf = gpd.read_file(f"{DATA_DIR}/index.gpkg")
    gdf = gdf.dissolve().to_crs(f"EPSG:{output_epsg}")
    logger.info(f"The approximate area is {gdf.area.values[0] / 1e+6:.1f} km2")
    # Create tile_dir
    tile_dir = f"{DATA_DIR}_tiles"
    if not os.path.exists(tile_dir):
        os.mkdir(tile_dir)
    if (len(files_to_process) > 1) | (
        (len(files_to_process) == 1) & (gdf.area.values[0] / 1e6 > 0.25) & (PTS_COUNT > 50000000)
    ):  # 25 ha or 50 mil points
        logger.info(f"Splitting point cloud into {tile_size}m tiles with {tile_buffer}m buffers ...")
        laz_tile(
            input_laz_files=f"{DATA_DIR}/*.laz",
            output_epsg=output_epsg,
            output_dir=tile_dir,
            length=tile_size,
            buffer=tile_buffer,
        )
    else:
        logger.info("Processing without tiling, as theres is just one file <25 ha or it contains <50 mil points...")
        shutil.copyfile(files_to_process[0], f"{tile_dir}/{os.path.basename(files_to_process[0])}")

    # Generate spatial index:
    logger.info("Generating spatial index for new tiles ...")
    laz_tindex(f"{tile_dir}")

    logger.info("Generating erosion heatmap ...")
    input_laz_files = glob(os.path.join(tile_dir, "*_grd.laz"))

    with ProcessPoolExecutor(cores) as pool:
        _pipeline = partial(
            erosion_pipeline,
            ext=".laz",
            output_epsg=output_epsg,
            gsd=GSD,
            buffer=tile_buffer / 2,  # to remove border artifacts
            nodata=nodata,
            smooth_radius=smooth_radius,
            height_thresh=z_threshold,
            scale_int16=False,
        )
        result = list(pool.map(_pipeline, input_laz_files))
        assert result

    output_erosion_basename = "erosion"
    output_dtm_basename = "dtm"
    logger.info("Merging Erosion Heatmap tiles ...")
    vrt_options = gdal.BuildVRTOptions(xRes=GSD, yRes=GSD, targetAlignedPixels=True)
    input_erosion_files = glob(os.path.join(tile_dir, "*_erosion_clean.tif"))
    gdal.BuildVRT(
        destName=f"{DATA_DIR}/{output_erosion_basename}.vrt",
        srcDSOrSrcDSTab=input_erosion_files,
        targetAlignedPixels=True,
        options=vrt_options,
    )

    kwargs = {
        "format": "GTiff",
        "outputType": gdal.GDT_Float32,
        "srcNodata": nodata,
        "dstNodata": nodata,
        "dstSRS": f"EPSG:{output_epsg}",
        "xRes": GSD,
        "yRes": GSD,
        "targetAlignedPixels": True,
        "creationOptions": [
            "COMPRESS=DEFLATE",
            "BIGTIFF=YES",
            "ZLEVEL=9",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "TILED=YES",
            "NUM_THREADS=ALL_CPUS",
        ],
    }
    gdal.Warp(
        destNameOrDestDS=f"{DATA_DIR}/{output_erosion_basename}.tif",
        srcDSOrSrcDSTab=f"{DATA_DIR}/{output_erosion_basename}.vrt",
        **kwargs,
    )

    ds = gdal.Open(f"{DATA_DIR}/{output_erosion_basename}.tif", gdal.GA_Update)
    ds.GetRasterBand(1).ComputeStatistics(0)
    del ds

    logger.info("Merging DTM tiles ...")
    input_dtm_files = glob(os.path.join(tile_dir, "*_dtm_clip.tif"))
    gdal.BuildVRT(
        destName=f"{DATA_DIR}/{output_dtm_basename}.vrt",
        srcDSOrSrcDSTab=input_dtm_files,
        targetAlignedPixels=True,
        options=vrt_options,
    )
    gdal.Warp(
        destNameOrDestDS=f"{DATA_DIR}/{output_dtm_basename}.tif",
        srcDSOrSrcDSTab=f"{DATA_DIR}/{output_dtm_basename}.vrt",
        **kwargs,
    )
    ds = gdal.Open(f"{DATA_DIR}/{output_dtm_basename}.tif", gdal.GA_Update)
    ds.GetRasterBand(1).ComputeStatistics(0)
    del ds

    logger.info("Generating Hillshade ...")
    kwargs = {
        "format": "GTiff",
        "computeEdges": True,
        "creationOptions": [
            "COMPRESS=DEFLATE",
            "BIGTIFF=YES",
            "ZLEVEL=9",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "TILED=YES",
            "NUM_THREADS=ALL_CPUS",
        ],
    }

    for processing in ["hillshade"]:  # "slope", "hillshade", "aspect", "color-relief", "TRI", "TPI", "Roughness"
        gdal.DEMProcessing(
            destName=f"{DATA_DIR}/{processing}.tif",
            srcDS=f"{DATA_DIR}/{output_dtm_basename}.tif",
            processing=processing,
            **kwargs,
        )

    logger.info("Reclassifying erosion into RAG classes and computing stats ...")
    reclassify_erosion(
        input_raster=f"{DATA_DIR}/{output_erosion_basename}.tif",
        output_raster=f"{DATA_DIR}/{output_erosion_basename.replace('erosion', 'erosion_class')}.tif",
        num_workers=cores,
        compute_json_stats=True,
        compute_raster_stats=True,
        build_overviews=False,
        nodata=nodata,
    )

    # Colourise erosion
    output_tif = os.path.splitext(os.path.basename(f"{DATA_DIR}/{output_erosion_basename}.tif"))[0] + ".COLOURISED.TIF"
    metadata_json = create_basic_colourmap(
        input_tif=f"{DATA_DIR}/{output_erosion_basename}.tif",
        output_tif=output_tif,
        colourmap=COLOURMAP_EROSION,
        legend_preset="elevation",
        legend_keys=[0.001, 0.3, 0.7, 1.1, 1.5],  # Needs to be consistent with erosion.gdaldem
    )

    # Copy outputs to s3
    output_prefix = f"{slugify(base_url)}/{job_id}/"

    # TIFs
    s3_copy_file(f"{DATA_DIR}/{output_erosion_basename}.tif", OUTPUT_BUCKET, output_prefix)
    s3_copy_file(output_tif, OUTPUT_BUCKET, output_prefix)
    s3_copy_file(metadata_json, OUTPUT_BUCKET, output_prefix)

    # Emit event to say processing is complete
    put_event(
        detail_type="LidarErosion:OutputData:Ready",
        detail={
            "job_id": job_id,
            "base_url": base_url,
            "output_bucket": OUTPUT_BUCKET,
            "output_prefix": output_prefix,
        },
    )
