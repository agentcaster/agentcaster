#!/usr/bin/env python3

import os
import glob
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECTION_PARAMS = {
    "proj": "lcc", 
    "lat_1": 25.0, 
    "lat_2": 25.0, 
    "lat_0": 25.0,
    "lon_0": -95.0, 
    "a": 6371200, 
    "b": 6371200, 
    "units": "m"
}
TARGET_CRS = pyproj.CRS(PROJECTION_PARAMS)
SOURCE_CRS = "EPSG:4326"

CONUS_LON_MIN, CONUS_LON_MAX = -121.0, -65.0
CONUS_LAT_MIN, CONUS_LAT_MAX = 22.0, 50.0

RISK_CATEGORIES = ['0%', '2%', '5%', '10%', '15%', '30%', '45%', '60%']
RISK_LEVEL_MAPPING = {
    '<2%': '2%', '0': '0%', '0.02': '2%', '0.05': '5%', '0.10': '10%', '0.15': '15%',
    '0.30': '30%', '0.45': '45%', '0.60': '60%', '2%': '2%', '5%': '5%', '10%': '10%',
    '15%': '15%', '30%': '30%', '45%': '45%', '60%': '60%'
}
RISK_WEIGHTS = {'0%': 1, '2%': 2, '5%': 5, '10%': 10, '15%': 15, '30%': 30, '45%': 45, '60%': 60}

RISK_COLORS = {
    '0%': '#FFFFFF', '2%': '#80c480', '5%': '#c6a393', '10%': '#ffeb80', '15%': '#ff8080',
    '30%': '#ff80ff', '45%': '#c896f7', '60%': '#0f4e8b'
}

CONTOUR_COLORS = {
    '2%': '#008200',
    '5%': '#8b4825',
    '10%': '#ff9601',
    '15%': '#ff0000',
    '30%': '#ff00ff',
    '45%': '#912dee',
    '60%': '#000000'
}


def canonicalize_label(raw):
    if raw is None:
        return '0%'
    try:
        key = str(raw).strip().lower()
        return RISK_LEVEL_MAPPING.get(key, '0%')
    except (TypeError, ValueError):
        return '0%'


def search_files_glob(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_domain_polygon():
    domain_box = box(CONUS_LON_MIN, CONUS_LAT_MIN, CONUS_LON_MAX, CONUS_LAT_MAX)
    
    gdf = gpd.GeoDataFrame(geometry=[domain_box], crs=SOURCE_CRS)
    
    gdf = gdf.to_crs(TARGET_CRS)
    
    return gdf.geometry.iloc[0]


def load_and_preprocess_prediction(filepath, target_crs):
    logger.debug(f"Loading prediction file: {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=target_crs)
    
    try:
        gdf = gpd.read_file(filepath)
        
        if gdf.empty:
            logger.warning(f"Empty file: {filepath}")
            return gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=target_crs)
        
        risk_column = None
        if 'risk_level' in gdf.columns:
            risk_column = 'risk_level'
        elif 'LABEL' in gdf.columns:
            risk_column = 'LABEL'
        elif 'DN' in gdf.columns:
            gdf['risk_level'] = gdf['DN'].apply(lambda x: f"{x/100:.2f}" if x > 0 else '0')
            risk_column = 'risk_level'
        
        if not risk_column:
            logger.warning(f"No risk level column found in {filepath}")
            return gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=target_crs)
        
        if risk_column != 'risk_level':
            gdf['risk_level'] = gdf[risk_column].apply(canonicalize_label)
        else:
            gdf['risk_level'] = gdf['risk_level'].apply(canonicalize_label)
        
        logger.debug(f"CRS before reprojection: {gdf.crs}")

        if gdf.crs is None:
            logger.warning(f"No CRS found in {filepath}, assuming WGS84")
            gdf.set_crs(SOURCE_CRS, inplace=True)
            logger.debug(f"CRS after assuming WGS84: {gdf.crs}")

        if gdf.crs != target_crs:
            logger.debug(f"Reprojecting from {gdf.crs} to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            logger.debug(f"CRS after reprojection: {gdf.crs}")
        else:
            logger.debug(f"CRS already matches target: {gdf.crs}")
        
        gdf['geometry'] = gdf['geometry'].buffer(0)
        
        processed_gdf = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
        
        higher_risk_geom = None
        
        for level in reversed(RISK_CATEGORIES[1:]):
            level_gdf = gdf[gdf['risk_level'] == level]
            
            if level_gdf.empty:
                continue
            
            try:
                level_geom = unary_union(level_gdf.geometry.values)
                
                if higher_risk_geom is not None and not higher_risk_geom.is_empty:
                    level_geom = level_geom.difference(higher_risk_geom)
                
                if level_geom and not level_geom.is_empty:
                    new_row = level_gdf.iloc[0:1].copy()
                    new_row['geometry'] = level_geom
                    new_row['risk_level'] = level
                    processed_gdf = pd.concat([processed_gdf, new_row], ignore_index=True)
                    
                    if higher_risk_geom is None:
                        higher_risk_geom = level_geom
                    else:
                        higher_risk_geom = unary_union([higher_risk_geom, level_geom])
            
            except Exception as e:
                logger.warning(f"Error processing risk level {level}: {e}")
                continue
        
        return processed_gdf
    
    except Exception as e:
        logger.error(f"Error loading prediction file {filepath}: {e}")
        return gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=target_crs)


def load_gt(date, target_crs):
    ppf_geojson = f"ppf_output/{date}/ground_truth_{date}.geojson"

    if not os.path.exists(ppf_geojson):
        logger.warning(f"Ground truth GeoJSON not found: {ppf_geojson}")
        return gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=target_crs)
    
    return load_and_preprocess_prediction(ppf_geojson, target_crs)


def load_llm(date, target_crs):
    llm_files = glob.glob(f"llm_predictions/{date}/prediction_*.geojson")
    
    if not llm_files:
        logger.warning(f"No LLM prediction files found for date {date}")
        return {}
    
    results = {}
    for llm_file in llm_files:
        try:
            filename = os.path.basename(llm_file)
            parts = filename.replace("prediction_", "").replace(".geojson", "").split(f"_{date}")
            model_name = parts[0] if len(parts) > 0 else f"unknown_{filename}"
            
            gdf = load_and_preprocess_prediction(llm_file, target_crs)
            results[model_name] = gdf
        
        except Exception as e:
            logger.error(f"Error loading LLM prediction {llm_file}: {e}")
            continue
    
    return results


def get_max_risk_level(gdf):
    if gdf.empty or 'risk_level' not in gdf.columns or gdf['risk_level'].isnull().all():
        return '0%'
    
    present_levels = gdf['risk_level'].dropna().unique()
    
    max_level = '0%'
    max_index = -1
    for level in present_levels:
        try:
            level_index = RISK_CATEGORIES.index(level)
            if level_index > max_index:
                max_index = level_index
                max_level = level
        except ValueError:
            logger.warning(f"Encountered unexpected risk level '{level}' while finding max risk.")
            continue
            
    return max_level


def load_spc(date, target_crs):
    base_dir = f"dataset/hrrr_{date}_00z/spc_outlooks"
    
    patterns = [
        f"{base_dir}/day1otlk_{date}_1200_torn.shp",
        f"{base_dir}/day1otlk_{date}_0[0-9]00_torn.shp",
        f"{base_dir}/day1otlk_{date}_1[0-9]00_torn.shp"
    ]
    
    spc_file = None
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            spc_file = matches[0]
            break
    
    if not spc_file:
        logger.warning(f"No SPC outlook file found for date {date}")
        return None
    
    return load_and_preprocess_prediction(spc_file, target_crs)


def compute_geometric_iou(gdf_gt, gdf_pred, domain_poly):
    ious = {}
    union_areas = {}
    
    for level in RISK_CATEGORIES[1:]:
        geom_gt = unary_union(gdf_gt[gdf_gt['risk_level'] == level].geometry.values) \
            if not gdf_gt[gdf_gt['risk_level'] == level].empty else Polygon()
            
        geom_pred = unary_union(gdf_pred[gdf_pred['risk_level'] == level].geometry.values) \
            if not gdf_pred[gdf_pred['risk_level'] == level].empty else Polygon()
        
        if geom_gt.is_empty and geom_pred.is_empty:
            ious[level] = 1.0
            union_areas[level] = 0.0
            continue
        
        try:
            intersection_geom = geom_gt.intersection(geom_pred)
            union_geom = geom_gt.union(geom_pred)
            
            intersection_area = intersection_geom.area
            union_area = union_geom.area
            
            iou = intersection_area / union_area if union_area > 1e-9 else 0.0
            ious[level] = float(iou)
            union_areas[level] = float(union_area)
        
        except Exception as e:
            logger.warning(f"Error calculating IoU for level {level}: {e}")
            ious[level] = 0.0
            union_areas[level] = 0.0
    
    try:
        geom_gt_nonzero = unary_union(gdf_gt.geometry.values) \
            if not gdf_gt.empty else Polygon()
            
        geom_pred_nonzero = unary_union(gdf_pred.geometry.values) \
            if not gdf_pred.empty else Polygon()
        
        geom_gt_zero = domain_poly.difference(geom_gt_nonzero)
        geom_pred_zero = domain_poly.difference(geom_pred_nonzero)
        
        intersection_zero = geom_gt_zero.intersection(geom_pred_zero)
        union_zero = geom_gt_zero.union(geom_pred_zero)
        
        intersection_area = intersection_zero.area
        union_area = union_zero.area
        
        iou_zero = intersection_area / union_area if union_area > 1e-9 else 1.0
        ious['0%'] = float(iou_zero)
        union_areas['0%'] = float(union_area)
    
    except Exception as e:
        logger.warning(f"Error calculating IoU for '0%' category: {e}")
        ious['0%'] = 0.0
        union_areas['0%'] = 0.0
    
    return ious, union_areas, geom_gt_nonzero, geom_pred_nonzero


def daily_scores(ious, union_areas, gdf_gt, gdf_pred, geom_gt_nonzero, geom_pred_nonzero, max_risk_pred):
    mean_iou = np.mean(list(ious.values())) if ious else 0.0

    is_quiet_day_gt = gdf_gt.empty or all(g.is_empty for g in gdf_gt.geometry)
    is_quiet_day_pred = gdf_pred.empty or all(g.is_empty for g in gdf_pred.geometry)

    if is_quiet_day_gt:
        tb_score = 1.0 if is_quiet_day_pred else 0.0
    else:
        relevant_levels = [level for level in RISK_CATEGORIES[1:] 
                           if level in ious and union_areas.get(level, 0.0) > 1e-9]
                           
        tb_score = np.mean([ious[level] for level in relevant_levels]) if relevant_levels else 0.0
    
    if is_quiet_day_gt:
        max_gt_risk = '0%'
    else:
        for level in reversed(RISK_CATEGORIES):
            if level in gdf_gt['risk_level'].values:
                max_gt_risk = level
                break
        else:
            max_gt_risk = '0%'
    
    weight = RISK_WEIGHTS[max_gt_risk]

    is_false_alarm = is_quiet_day_gt and not is_quiet_day_pred

    is_risk_day_hallucination = False
    if not is_quiet_day_gt and not is_quiet_day_pred:
        try:
            intersection_area = geom_gt_nonzero.intersection(geom_pred_nonzero).area
            if intersection_area < 1e-9:
                is_risk_day_hallucination = True
        except Exception as e:
            logger.warning(f"Error calculating intersection for hard hallucination check: {e}")
            is_risk_day_hallucination = False

    is_hard_hallucination = is_false_alarm or is_risk_day_hallucination

    daily_weighted_hallucination_loss = 0.0
    if is_hard_hallucination:
        predicted_weight = RISK_WEIGHTS.get(max_risk_pred, 1)
        daily_weighted_hallucination_loss = float(predicted_weight)

    return mean_iou, tb_score, weight, is_false_alarm, is_hard_hallucination, daily_weighted_hallucination_loss


def plot_vector_comparison(gdf_gt, gdf_pred, date, model, target_crs, tornadobench_score, gt_overall_centroid_coords=None, pred_overall_centroid_coords=None, gt_hr_centroid_coords=None, pred_hr_centroid_coords=None):
    output_path = f"iou_results/plots/iou_comparison_{model}_{date}.png"
    
    try:
        map_proj = ccrs.LambertConformal(
            central_longitude=PROJECTION_PARAMS['lon_0'],
            central_latitude=PROJECTION_PARAMS['lat_0'],
            standard_parallels=(PROJECTION_PARAMS['lat_1'], PROJECTION_PARAMS['lat_2']),
            globe=ccrs.Globe(ellipse=None, semimajor_axis=PROJECTION_PARAMS['a'], semiminor_axis=PROJECTION_PARAMS['b'])
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.6), 
                                      subplot_kw={'projection': map_proj})
        
        proj_latlon = pyproj.Proj("epsg:4326")
        proj_map = pyproj.Proj(map_proj.proj4_params)
        transformer_ll_to_map = pyproj.Transformer.from_proj(proj_latlon, proj_map, always_xy=True)
        
        min_x, min_y = transformer_ll_to_map.transform(CONUS_LON_MIN, CONUS_LAT_MIN)
        max_x, max_y = transformer_ll_to_map.transform(CONUS_LON_MAX, CONUS_LAT_MAX)
        
        map_bounds_poly = box(min_x, min_y, max_x, max_y)
        
        for ax in [ax1, ax2]:
            ax.set_extent([min_x, max_x, min_y, max_y], crs=map_proj)
            
            ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#d3e4f5')
            ax.add_feature(cfeature.LAND, zorder=0, facecolor='#ffffff')
            ax.add_feature(cfeature.LAKES, zorder=1, alpha=0.5, facecolor='#d3e4f5')
            ax.add_feature(cfeature.COASTLINE, zorder=3, edgecolor='#949494', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, zorder=3, linestyle='-', edgecolor='#949494', linewidth=0.5)
            ax.add_feature(cfeature.STATES, zorder=3, linestyle='-', edgecolor='#949494', linewidth=0.3)
        
        for level in RISK_CATEGORIES[1:]:
            level_gt_data = gdf_gt[gdf_gt['risk_level'] == level]
            if not level_gt_data.empty:
                geoms_to_plot_gt = []
                for geom in level_gt_data.geometry:
                    if geom.is_empty:
                        continue
                    if geom.geom_type == 'Polygon':
                        geoms_to_plot_gt.append(geom)
                    elif geom.geom_type == 'MultiPolygon':
                        geoms_to_plot_gt.extend(list(geom.geoms))
                    elif geom.geom_type == 'GeometryCollection':
                        for sub_geom in geom.geoms:
                            if sub_geom.geom_type == 'Polygon':
                                geoms_to_plot_gt.append(sub_geom)
                            elif sub_geom.geom_type == 'MultiPolygon':
                                geoms_to_plot_gt.extend(list(sub_geom.geoms))
                
                if geoms_to_plot_gt:
                    ax1.add_geometries(
                        geoms_to_plot_gt,
                        crs=ccrs.LambertConformal(
                            central_longitude=PROJECTION_PARAMS['lon_0'],
                            central_latitude=PROJECTION_PARAMS['lat_0'],
                            standard_parallels=(PROJECTION_PARAMS['lat_1'], PROJECTION_PARAMS['lat_2'])
                        ),
                        facecolor=RISK_COLORS[level],
                        edgecolor=CONTOUR_COLORS.get(level, '#000000'),
                        linewidth=0.5,
                        alpha=1.0,
                        zorder=2
                    )
        
        for level in RISK_CATEGORIES[1:]:
            level_pred_data = gdf_pred[gdf_pred['risk_level'] == level]
            if not level_pred_data.empty:
                geoms_to_plot_pred = []
                for geom in level_pred_data.geometry:
                    if geom.is_empty:
                        continue
                    if geom.geom_type == 'Polygon':
                        geoms_to_plot_pred.append(geom)
                    elif geom.geom_type == 'MultiPolygon':
                        geoms_to_plot_pred.extend(list(geom.geoms))
                    elif geom.geom_type == 'GeometryCollection':
                        for sub_geom in geom.geoms:
                            if sub_geom.geom_type == 'Polygon':
                                geoms_to_plot_pred.append(sub_geom)
                            elif sub_geom.geom_type == 'MultiPolygon':
                                geoms_to_plot_pred.extend(list(sub_geom.geoms))
                
                if geoms_to_plot_pred:
                    ax2.add_geometries(
                        geoms_to_plot_pred,
                        crs=ccrs.LambertConformal(
                            central_longitude=PROJECTION_PARAMS['lon_0'],
                            central_latitude=PROJECTION_PARAMS['lat_0'],
                            standard_parallels=(PROJECTION_PARAMS['lat_1'], PROJECTION_PARAMS['lat_2'])
                        ),
                        facecolor=RISK_COLORS[level],
                        edgecolor=CONTOUR_COLORS.get(level, '#000000'),
                        linewidth=0.5,
                        alpha=1.0,
                        zorder=2
                    )
        
        plot_projection = map_proj
        
        for level in RISK_CATEGORIES[1:]:
            geom_gt_level = unary_union(gdf_gt[gdf_gt['risk_level'] == level].geometry.values) \
                if not gdf_gt[gdf_gt['risk_level'] == level].empty else Polygon()
            geom_pred_level = unary_union(gdf_pred[gdf_pred['risk_level'] == level].geometry.values) \
                if not gdf_pred[gdf_pred['risk_level'] == level].empty else Polygon()

            if not geom_gt_level.is_valid: geom_gt_level = geom_gt_level.buffer(0)
            if not geom_pred_level.is_valid: geom_pred_level = geom_pred_level.buffer(0)

            try:
                intersection_geom = geom_gt_level.intersection(geom_pred_level)
                
                if intersection_geom and not intersection_geom.is_empty:
                    if not intersection_geom.is_valid: intersection_geom = intersection_geom.buffer(0) 
                    
                    clipped_intersection_geom = intersection_geom.intersection(map_bounds_poly)
                    
                    if clipped_intersection_geom and not clipped_intersection_geom.is_empty:
                        if not clipped_intersection_geom.is_valid: clipped_intersection_geom = clipped_intersection_geom.buffer(0)
                        
                        geoms_to_plot_intersection = []
                        if clipped_intersection_geom.geom_type == 'Polygon':
                            geoms_to_plot_intersection.append(clipped_intersection_geom)
                        elif clipped_intersection_geom.geom_type == 'MultiPolygon':
                            geoms_to_plot_intersection.extend([p for p in clipped_intersection_geom.geoms if p.is_valid and not p.is_empty])
                        elif clipped_intersection_geom.geom_type == 'GeometryCollection':
                            for sub_geom in clipped_intersection_geom.geoms:
                                if sub_geom.geom_type == 'Polygon' and sub_geom.is_valid and not sub_geom.is_empty:
                                    geoms_to_plot_intersection.append(sub_geom)
                                elif sub_geom.geom_type == 'MultiPolygon':
                                    geoms_to_plot_intersection.extend([p for p in sub_geom.geoms if p.is_valid and not p.is_empty])
                    
                        if geoms_to_plot_intersection:
                             for ax in [ax1, ax2]:
                                  valid_geoms_for_plotting = [g for g in geoms_to_plot_intersection if g.is_valid and not g.is_empty]
                                  if valid_geoms_for_plotting:
                                      ax.add_geometries(
                                          valid_geoms_for_plotting,
                                          crs=plot_projection,
                                          facecolor='black',
                                          edgecolor='none',
                                          alpha=0.4,
                                          zorder=4
                                      )
            except Exception as e:
                 logger.warning(f"Could not calculate or plot intersection for level {level}: {e}")
                 continue

        ax1.set_title("Ground Truth")
        ax2.set_title(f"Prediction - {model}")
        
        if gt_overall_centroid_coords and gt_overall_centroid_coords[0] is not None and gt_overall_centroid_coords[1] is not None:
            ax1.plot(gt_overall_centroid_coords[0], gt_overall_centroid_coords[1], 'bv', markersize=6, markeredgewidth=1.0, transform=map_proj, label='GT Overall Centroid', zorder=5)
            
        if pred_overall_centroid_coords and pred_overall_centroid_coords[0] is not None and pred_overall_centroid_coords[1] is not None:
            ax2.plot(pred_overall_centroid_coords[0], pred_overall_centroid_coords[1], 'bv', markersize=6, markeredgewidth=1.0, transform=map_proj, label='Pred Overall Centroid', zorder=5)

        if gt_hr_centroid_coords and gt_hr_centroid_coords[0] is not None and gt_hr_centroid_coords[1] is not None:
            ax1.plot(gt_hr_centroid_coords[0], gt_hr_centroid_coords[1], 'rv', markersize=6, markeredgewidth=1.0, transform=map_proj, label='HR Centroid', zorder=6)
        if pred_hr_centroid_coords and pred_hr_centroid_coords[0] is not None and pred_hr_centroid_coords[1] is not None:
            ax2.plot(pred_hr_centroid_coords[0], pred_hr_centroid_coords[1], 'rv', markersize=6, markeredgewidth=1.0, transform=map_proj, label='HR Centroid', zorder=6)

        legend_patches = []
        for level in RISK_CATEGORIES[1:]:
            patch = mpatches.Patch(color=RISK_COLORS[level], label=level)
            legend_patches.append(patch)
        
        date_patch = mpatches.Patch(color='white', label=f"Date: {date}", alpha=0)
        score_patch = mpatches.Patch(color='white', label=f"Score: {tornadobench_score*100:.2f}%", alpha=0)
        legend_patches.append(date_patch)
        legend_patches.append(score_patch)

        gt_overall_centroid_patch = mlines.Line2D([], [], color='blue', marker='v', linestyle='None', markersize=5, label='Overall Risk Centroid')
        legend_patches.append(gt_overall_centroid_patch)
        hr_centroid_patch = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=5, label='Max Risk Centroid')
        legend_patches.append(hr_centroid_patch)
        
        fig.legend(
            handles=legend_patches,
            loc='lower center',
            ncol=len(legend_patches),
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
            borderaxespad=0.02,
            handletextpad=0.3,
            columnspacing=0.8
        )

        plt.subplots_adjust(bottom=0.05, wspace=0.02, top=0.95, left=0.02, right=0.98)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved comparison plot to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")


def plot_summary_bars(df_summary):
    try:
        ensure_dir("iou_results/plots")
        
        df_plot = df_summary.set_index('model')
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.axisbelow': True,
            'axes.edgecolor': '#444444',
            'axes.linewidth': 0.8
        })
        
        color_tb = '#e74c3c'
        color_ths = '#2ecc71'
        color_thh = '#9b59b6'
        
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
        
        series_tb = df_plot['TornadoBench'].sort_values(ascending=False) 
        
        model_names = series_tb.index.tolist()
        display_names = [name.split('_', 1)[1] if '_' in name else name for name in model_names]
        
        x_pos = np.arange(len(model_names))
        
        bars2 = ax2.bar(x_pos, series_tb.values, width=0.7, color=color_tb, 
                       edgecolor='none', alpha=0.85)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.add_patch(plt.Rectangle((x_pos[i]-0.35+0.02, 0), 0.7, height, fill=True, 
                                      color='black', alpha=0.05, zorder=0))
        
        ax2.set_ylabel('TornadoBench Score (%)', fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_title('TornadoBench', pad=15)
        
        ax2.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3, color='#888888')
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(0.5)
        ax2.spines['bottom'].set_linewidth(0.5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}%',
                        xy=(x_pos[i], height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='#444444')
        
        fig2.patch.set_facecolor('#f8f9fa')
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])
        plt.savefig("iou_results/model_summary_TornadoBench.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        if 'TornadoHallucinationSimple' in df_plot.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)

            series_ths = df_plot['TornadoHallucinationSimple'].sort_values(ascending=True)

            model_names = series_ths.index.tolist()
            display_names = [name.split('_', 1)[1] if '_' in name else name for name in model_names]

            x_pos = np.arange(len(model_names))

            bars3 = ax3.bar(x_pos, series_ths.values, width=0.7, color=color_ths,
                          edgecolor='none', alpha=0.85)

            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax3.add_patch(plt.Rectangle((x_pos[i]-0.35+0.02, 0), 0.7, height, fill=True,
                                         color='black', alpha=0.05, zorder=0))

            ax3.set_ylabel('TornadoHallucinationSimple', fontweight='bold')
            ax3.set_xlabel('')
            ax3.set_title('TornadoHallucinationSimple (Lower is Better)', pad=15)
            ax3.set_ylim(0, min(1.05, max(series_ths.values) * 1.1))

            ax3.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3, color='#888888')

            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_linewidth(0.5)
            ax3.spines['bottom'].set_linewidth(0.5)

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(display_names, rotation=45, ha='right')

            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax3.annotate(f'{height:.2f}',
                           xy=(x_pos[i], height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           color='#444444')

            fig3.patch.set_facecolor('#f8f9fa')
            ax3.set_facecolor('#f8f9fa')

            plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])
            plt.savefig("iou_results/model_summary_TornadoHallucinationSimple.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
            logger.info("Saved TornadoHallucinationSimple bar plot")
        else:
            logger.warning("'TornadoHallucinationSimple' column not found in summary data, skipping plot.")

        if 'TornadoHallucinationHard' in df_plot.columns:
            fig5, ax5 = plt.subplots(figsize=(10, 6), dpi=300)

            series_whl = df_plot['TornadoHallucinationHard'].sort_values(ascending=True)

            model_names = series_whl.index.tolist()
            display_names = [name.split('_', 1)[1] if '_' in name else name for name in model_names]

            x_pos = np.arange(len(model_names))

            bars5 = ax5.bar(x_pos, series_whl.values, width=0.7, color=color_thh,
                          edgecolor='none', alpha=0.85)

            for i, bar in enumerate(bars5):
                height = bar.get_height()
                ax5.add_patch(plt.Rectangle((x_pos[i]-0.35+0.02, 0), 0.7, height, fill=True,
                                         color='black', alpha=0.05, zorder=0))

            ax5.set_ylabel('TornadoHallucinationHard', fontweight='bold')
            ax5.set_xlabel('')
            ax5.set_title('TornadoHallucinationHard (Lower is Better)', pad=15)
            
            y_max = max(series_whl.values) * 1.1
            ax5.set_ylim(0, y_max)

            ax5.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3, color='#888888')

            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['left'].set_linewidth(0.5)
            ax5.spines['bottom'].set_linewidth(0.5)

            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(display_names, rotation=45, ha='right')

            for i, bar in enumerate(bars5):
                height = bar.get_height()
                ax5.annotate(f'{height:.2f}',
                           xy=(x_pos[i], height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           color='#444444')

            fig5.patch.set_facecolor('#f8f9fa')
            ax5.set_facecolor('#f8f9fa')

            plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.95])
            plt.savefig("iou_results/model_summary_TornadoHallucinationHard.png", dpi=300, bbox_inches='tight')
            plt.close(fig5)
            logger.info("Saved TornadoHallucinationHard bar plot")
        else:
            logger.warning("'TornadoHallucinationHard' column not found in summary data, skipping plot.")

        logger.info("Saved bar plots")
    
    except Exception as e:
        logger.error(f"Error creating summary bar plots: {e}")
        import traceback
        logger.error(traceback.format_exc())


def plot_max_risk_match_summary(df_summary):
    required_cols = ['model', 'UnderforecastRate', 'MatchRate', 'OverforecastRate']
    if not all(col in df_summary.columns for col in required_cols):
        logger.error(f"Missing required columns for Max Risk Match plot. Need: {required_cols}")
        return

    try:
        df_plot = df_summary[required_cols].copy()
        df_plot = df_plot.set_index('model')

        df_plot.sort_values(by=['MatchRate', 'OverforecastRate'], ascending=[False, True], inplace=True)

        color_under = '#3498db'
        color_match = '#2ecc71'
        color_over = '#e74c3c'

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        model_names = df_plot.index.tolist()
        display_names = [name.split('_', 1)[1] if '_' in name else name for name in model_names]
        x_pos = np.arange(len(model_names))

        bars_under = ax.bar(x_pos, df_plot['UnderforecastRate'], width=0.7,
                           label='Underforecast', color=color_under, edgecolor='none', alpha=0.85)
        bars_match = ax.bar(x_pos, df_plot['MatchRate'], width=0.7,
                           label='Match', color=color_match, edgecolor='none', alpha=0.85,
                           bottom=df_plot['UnderforecastRate'])
        bars_over = ax.bar(x_pos, df_plot['OverforecastRate'], width=0.7,
                          label='Overforecast', color=color_over, edgecolor='none', alpha=0.85,
                          bottom=df_plot['UnderforecastRate'] + df_plot['MatchRate'])

        ax.set_ylabel('Percentage of Days (%)', fontweight='bold')
        ax.set_xlabel('')
        ax.set_title('Maximum Risk Forecast Performance vs. Ground Truth', pad=15)
        ax.set_ylim(0, 100)

        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3, color='#888888')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.tick_params(axis='x', length=0)

        ax.legend(title="Max Risk Comparison", loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 0.9])
        output_path = "iou_results/model_summary_MaxRiskMatch.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Max Risk Match summary plot to {output_path}")

    except Exception as e:
        logger.error(f"Error creating Max Risk Match summary plot: {e}")
        import traceback
        logger.error(traceback.format_exc())


def plot_temporal_consistency(df_results):
    output_path = "iou_results/plots/temporal_performance_trends.png"
    ensure_dir("iou_results/plots")

    try:
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

        plot_data_full = df_results.copy()
        
        if 'gt_had_actual_risk' in plot_data_full.columns:
            plot_data = plot_data_full[plot_data_full['gt_had_actual_risk'] == True].copy()
            plot_title = "Temporal Consistency: Daily TornadoBench Scores (GT Risk Days Only)"
        else:
            logger.warning("'gt_had_actual_risk' column not found in results. Plotting all days for temporal consistency.")
            plot_data = plot_data_full
            plot_title = "Temporal Consistency: Daily TornadoBench Scores (All Days)"

        if plot_data.empty:
            logger.info("No data with ground truth risk to plot for temporal consistency. Skipping plot.")
            plt.close(fig)
            return

        plot_data['date'] = pd.to_datetime(plot_data['date'], format='%Y%m%d')

        models = sorted(plot_data['model'].unique())
        
        num_models = len(models)
        colors = plt.colormaps.get_cmap('tab10') if num_models <= 10 else plt.colormaps.get_cmap('tab20')

        for i, model in enumerate(models):
            model_data = plot_data[plot_data['model'] == model].sort_values(by='date')
            
            display_name = model.split('_', 1)[1] if '_' in model else model

            ax.plot(model_data['date'], model_data['daily_tornadobench'] * 100,
                    label=display_name, marker='o', linestyle='-', markersize=5, color=colors(i % colors.N), linewidth=1.5)

        ax.set_xlabel("Date", fontweight='bold', fontsize=14)
        ax.set_ylabel("Daily TornadoBench Score (%)", fontweight='bold', fontsize=14)
        ax.set_title(plot_title, pad=20, fontweight='bold', fontsize=18)
        
        ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, color='#888888')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=10, maxticks=40))
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        ax.legend(title="Model", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., fontsize=10, title_fontsize=12)

        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout(rect=[0, 0, 0.83, 1])
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved temporal consistency plot to {output_path}")

    except Exception as e:
        logger.error(f"Error creating temporal consistency plot: {e}")
        import traceback
        logger.error(traceback.format_exc())


def plot_vector_comparison_triple(gdf_gt, gdf_spc, gdf_model, date, model_name_spc, model_name_model, target_crs, tornadobench_score_spc, tornadobench_score_model, gt_overall_centroid_coords=None, spc_overall_centroid_coords=None, model_overall_centroid_coords=None, gt_hr_centroid_coords=None, spc_hr_centroid_coords=None, model_hr_centroid_coords=None):
    output_path = f"iou_results/plots/iou_comparison_triple_{date}_{model_name_model.replace('/', '_')}.png"
    
    try:
        map_proj = ccrs.LambertConformal(
            central_longitude=PROJECTION_PARAMS['lon_0'],
            central_latitude=PROJECTION_PARAMS['lat_0'],
            standard_parallels=(PROJECTION_PARAMS['lat_1'], PROJECTION_PARAMS['lat_2']),
            globe=ccrs.Globe(ellipse=None, semimajor_axis=PROJECTION_PARAMS['a'], semiminor_axis=PROJECTION_PARAMS['b'])
        )
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5.6), 
                                           subplot_kw={'projection': map_proj})
        
        proj_latlon = pyproj.Proj("epsg:4326")
        proj_map = pyproj.Proj(map_proj.proj4_params)
        transformer_ll_to_map = pyproj.Transformer.from_proj(proj_latlon, proj_map, always_xy=True)
        
        min_x, min_y = transformer_ll_to_map.transform(CONUS_LON_MIN, CONUS_LAT_MIN)
        max_x, max_y = transformer_ll_to_map.transform(CONUS_LON_MAX, CONUS_LAT_MAX)
        map_bounds_poly = box(min_x, min_y, max_x, max_y)
        
        axes = [ax1, ax2, ax3]
        for ax_idx, ax in enumerate(axes):
            ax.set_extent([min_x, max_x, min_y, max_y], crs=map_proj)
            ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#d3e4f5')
            ax.add_feature(cfeature.LAND, zorder=0, facecolor='#ffffff')
            ax.add_feature(cfeature.LAKES, zorder=1, alpha=0.5, facecolor='#d3e4f5')
            ax.add_feature(cfeature.COASTLINE, zorder=3, edgecolor='#949494', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, zorder=3, linestyle='-', edgecolor='#949494', linewidth=0.5)
            ax.add_feature(cfeature.STATES, zorder=3, linestyle='-', edgecolor='#949494', linewidth=0.3)

        for level in RISK_CATEGORIES[1:]:
            level_gt_data = gdf_gt[gdf_gt['risk_level'] == level]
            if not level_gt_data.empty:
                geoms_to_plot = []
                for geom in level_gt_data.geometry:
                    if geom.is_empty: continue
                    if geom.geom_type == 'Polygon': geoms_to_plot.append(geom)
                    elif geom.geom_type == 'MultiPolygon': geoms_to_plot.extend(list(geom.geoms))
                    elif geom.geom_type == 'GeometryCollection':
                        for sub_geom in geom.geoms:
                            if sub_geom.geom_type == 'Polygon': geoms_to_plot.append(sub_geom)
                            elif sub_geom.geom_type == 'MultiPolygon': geoms_to_plot.extend(list(sub_geom.geoms))
                if geoms_to_plot:
                    ax1.add_geometries(geoms_to_plot, crs=map_proj, facecolor=RISK_COLORS[level], edgecolor=CONTOUR_COLORS.get(level, '#000000'), linewidth=0.5, alpha=1.0, zorder=2)

        if gdf_spc is not None and not gdf_spc.empty:
            for level in RISK_CATEGORIES[1:]:
                level_spc_data = gdf_spc[gdf_spc['risk_level'] == level]
                if not level_spc_data.empty:
                    geoms_to_plot_spc = []
                    for geom in level_spc_data.geometry:
                        if geom.is_empty: continue
                        if geom.geom_type == 'Polygon': geoms_to_plot_spc.append(geom)
                        elif geom.geom_type == 'MultiPolygon': geoms_to_plot_spc.extend(list(geom.geoms))
                        elif geom.geom_type == 'GeometryCollection':
                            for sub_geom in geom.geoms:
                                if sub_geom.geom_type == 'Polygon': geoms_to_plot_spc.append(sub_geom)
                                elif sub_geom.geom_type == 'MultiPolygon': geoms_to_plot_spc.extend(list(sub_geom.geoms))
                    if geoms_to_plot_spc:
                        ax2.add_geometries(geoms_to_plot_spc, crs=map_proj, facecolor=RISK_COLORS[level], edgecolor=CONTOUR_COLORS.get(level, '#000000'), linewidth=0.5, alpha=1.0, zorder=2)
            for level in RISK_CATEGORIES[1:]:
                geom_gt_level = unary_union(gdf_gt[gdf_gt['risk_level'] == level].geometry.values) if not gdf_gt[gdf_gt['risk_level'] == level].empty else Polygon()
                geom_spc_level = unary_union(gdf_spc[gdf_spc['risk_level'] == level].geometry.values) if not gdf_spc[gdf_spc['risk_level'] == level].empty else Polygon()
                if not geom_gt_level.is_valid: geom_gt_level = geom_gt_level.buffer(0)
                if not geom_spc_level.is_valid: geom_spc_level = geom_spc_level.buffer(0)
                try:
                    intersection_geom = geom_gt_level.intersection(geom_spc_level)
                    if intersection_geom and not intersection_geom.is_empty:
                        if not intersection_geom.is_valid: intersection_geom = intersection_geom.buffer(0)
                        clipped_intersection_geom = intersection_geom.intersection(map_bounds_poly)
                        if clipped_intersection_geom and not clipped_intersection_geom.is_empty:
                            geoms_to_plot_intersection = []
                            if clipped_intersection_geom.geom_type == 'Polygon': geoms_to_plot_intersection.append(clipped_intersection_geom)
                            elif clipped_intersection_geom.geom_type == 'MultiPolygon': geoms_to_plot_intersection.extend([p for p in clipped_intersection_geom.geoms if p.is_valid and not p.is_empty])
                            elif clipped_intersection_geom.geom_type == 'GeometryCollection':
                                for sub_geom in clipped_intersection_geom.geoms:
                                    if sub_geom.geom_type == 'Polygon' and sub_geom.is_valid and not sub_geom.is_empty: geoms_to_plot_intersection.append(sub_geom)
                                    elif sub_geom.geom_type == 'MultiPolygon': geoms_to_plot_intersection.extend([p for p in sub_geom.geoms if p.is_valid and not p.is_empty])
                            if geoms_to_plot_intersection:
                                valid_geoms_for_plotting = [g for g in geoms_to_plot_intersection if g.is_valid and not g.is_empty]
                                if valid_geoms_for_plotting:
                                    ax2.add_geometries(valid_geoms_for_plotting, crs=map_proj, facecolor='black', edgecolor='none', alpha=0.4, zorder=4)
                except Exception as e: logger.warning(f"Could not calculate or plot GT-SPC intersection for level {level}: {e}")

        if gdf_model is not None and not gdf_model.empty:
            for level in RISK_CATEGORIES[1:]:
                level_model_data = gdf_model[gdf_model['risk_level'] == level]
                if not level_model_data.empty:
                    geoms_to_plot_model = []
                    for geom in level_model_data.geometry:
                        if geom.is_empty: continue
                        if geom.geom_type == 'Polygon': geoms_to_plot_model.append(geom)
                        elif geom.geom_type == 'MultiPolygon': geoms_to_plot_model.extend(list(geom.geoms))
                        elif geom.geom_type == 'GeometryCollection':
                            for sub_geom in geom.geoms:
                                if sub_geom.geom_type == 'Polygon': geoms_to_plot_model.append(sub_geom)
                                elif sub_geom.geom_type == 'MultiPolygon': geoms_to_plot_model.extend(list(sub_geom.geoms))

                    if geoms_to_plot_model:
                        ax3.add_geometries(geoms_to_plot_model, crs=map_proj, facecolor=RISK_COLORS[level], edgecolor=CONTOUR_COLORS.get(level, '#000000'), linewidth=0.5, alpha=1.0, zorder=2)
            for level in RISK_CATEGORIES[1:]:
                geom_gt_level = unary_union(gdf_gt[gdf_gt['risk_level'] == level].geometry.values) if not gdf_gt[gdf_gt['risk_level'] == level].empty else Polygon()
                geom_model_level = unary_union(gdf_model[gdf_model['risk_level'] == level].geometry.values) if not gdf_model[gdf_model['risk_level'] == level].empty else Polygon()
                if not geom_gt_level.is_valid: geom_gt_level = geom_gt_level.buffer(0)
                if not geom_model_level.is_valid: geom_model_level = geom_model_level.buffer(0)
                try:
                    intersection_geom = geom_gt_level.intersection(geom_model_level)
                    if intersection_geom and not intersection_geom.is_empty:
                        if not intersection_geom.is_valid: intersection_geom = intersection_geom.buffer(0)
                        clipped_intersection_geom = intersection_geom.intersection(map_bounds_poly)
                        if clipped_intersection_geom and not clipped_intersection_geom.is_empty:
                            geoms_to_plot_intersection = []
                            if clipped_intersection_geom.geom_type == 'Polygon': geoms_to_plot_intersection.append(clipped_intersection_geom)
                            elif clipped_intersection_geom.geom_type == 'MultiPolygon': geoms_to_plot_intersection.extend([p for p in clipped_intersection_geom.geoms if p.is_valid and not p.is_empty])
                            elif clipped_intersection_geom.geom_type == 'GeometryCollection':
                                for sub_geom in clipped_intersection_geom.geoms:
                                    if sub_geom.geom_type == 'Polygon' and sub_geom.is_valid and not sub_geom.is_empty: geoms_to_plot_intersection.append(sub_geom)
                                    elif sub_geom.geom_type == 'MultiPolygon': geoms_to_plot_intersection.extend([p for p in sub_geom.geoms if p.is_valid and not p.is_empty])
                            if geoms_to_plot_intersection:
                                valid_geoms_for_plotting = [g for g in geoms_to_plot_intersection if g.is_valid and not g.is_empty]
                                if valid_geoms_for_plotting:
                                    ax3.add_geometries(valid_geoms_for_plotting, crs=map_proj, facecolor='black', edgecolor='none', alpha=0.4, zorder=4)
                except Exception as e: logger.warning(f"Could not calculate or plot GT-Model intersection for level {level}: {e}")

        ax1.set_title("Ground Truth", fontsize=16)
        ax2.set_title(f"Prediction - {model_name_spc}", fontsize=16)
        ax3.set_title(f"Prediction - {model_name_model}", fontsize=16)

        if gt_overall_centroid_coords and gt_overall_centroid_coords[0] is not None:
            ax1.plot(gt_overall_centroid_coords[0], gt_overall_centroid_coords[1], 'bv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=5)
        if gt_hr_centroid_coords and gt_hr_centroid_coords[0] is not None:
            ax1.plot(gt_hr_centroid_coords[0], gt_hr_centroid_coords[1], 'rv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=6)

        if spc_overall_centroid_coords and spc_overall_centroid_coords[0] is not None:
            ax2.plot(spc_overall_centroid_coords[0], spc_overall_centroid_coords[1], 'bv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=5)
        if spc_hr_centroid_coords and spc_hr_centroid_coords[0] is not None:
            ax2.plot(spc_hr_centroid_coords[0], spc_hr_centroid_coords[1], 'rv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=6)

        if model_overall_centroid_coords and model_overall_centroid_coords[0] is not None:
            ax3.plot(model_overall_centroid_coords[0], model_overall_centroid_coords[1], 'bv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=5)
        if model_hr_centroid_coords and model_hr_centroid_coords[0] is not None:
            ax3.plot(model_hr_centroid_coords[0], model_hr_centroid_coords[1], 'rv', markersize=6, markeredgewidth=1.0, transform=map_proj, zorder=6)
        
        legend_patches = []
        for level in RISK_CATEGORIES[1:]:
            patch = mpatches.Patch(color=RISK_COLORS[level], label=level)
            legend_patches.append(patch)
        
        date_patch = mpatches.Patch(color='white', label=f"Date: {date}", alpha=0)
        legend_patches.append(date_patch)
        
        score_spc_patch = mpatches.Patch(color='white', label=f"SPC Score: {tornadobench_score_spc*100:.2f}%", alpha=0)
        legend_patches.append(score_spc_patch)
        score_model_patch = mpatches.Patch(color='white', label=f"Model Score: {tornadobench_score_model*100:.2f}%", alpha=0)
        legend_patches.append(score_model_patch)

        gt_overall_centroid_patch = mlines.Line2D([], [], color='blue', marker='v', linestyle='None', markersize=5, label='Overall Risk Centroid')
        legend_patches.append(gt_overall_centroid_patch)
        hr_centroid_patch = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=5, label='Max Risk Centroid')
        legend_patches.append(hr_centroid_patch)
        
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches), fontsize=16, frameon=True, bbox_to_anchor=(0.5, 0.01), borderaxespad=0.5, handletextpad=0.3, columnspacing=0.8)
        
        plt.subplots_adjust(bottom=0.15, wspace=0.005, top=0.95, left=0.01, right=0.99)

        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved triple comparison plot to {output_path}")
        plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error generating triple comparison plot for {date} and model {model_name_model}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Calculate IoU metrics for tornado predictions.")
    parser.add_argument("--start", help="Start date (YYYYMMDD)", type=str)
    parser.add_argument("--end", help="End date (YYYYMMDD)", type=str)
    parser.add_argument("--models", help="Comma-separated list of model names to evaluate", type=str)
    parser.add_argument("--skip_plots", help="Skip generating plots", action="store_true")
    parser.add_argument("--debug", help="Enable debug logging", action="store_true")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ensure_dir("iou_results")
    ensure_dir("iou_results/jsons")
    if not args.skip_plots:
        ensure_dir("iou_results/plots")
    
    logger.info("Creating domain polygon...")
    domain_polygon = create_domain_polygon()
    
    if args.start is None or args.end is None:
        date_dirs = sorted(glob.glob("ppf_output/????????"))
        available_dates = [os.path.basename(d) for d in date_dirs]
        
        if not available_dates:
            logger.error("No data found in ppf_output directory")
            return
        
        start_date = args.start or available_dates[0]
        end_date = args.end or available_dates[-1]
    else:
        start_date = args.start
        end_date = args.end
    
    logger.info(f"Processing dates from {start_date} to {end_date}")
    
    date_dirs = sorted(glob.glob("ppf_output/????????"))
    available_dates = [os.path.basename(d) for d in date_dirs 
                      if start_date <= os.path.basename(d) <= end_date]
    
    model_filter = None
    if args.models:
        model_filter = args.models.split(',')
        logger.info(f"Filtering to models: {model_filter}")
    
    all_results = []
    
    paper_submission_cases = {
        "20250314": "anthropic_claude-3.7-sonnet",
        "20250402": "anthropic_claude-3.7-sonnet_soundings_0"
    }

    for date in tqdm(available_dates, desc="Processing dates"):
        logger.info(f"Processing date: {date}")
        
        gt_data_for_triple_plot = {"gdf": None, "overall_coords": None, "hr_coords": None}
        spc_data_for_triple_plot = None
        target_model_data_for_triple_plot = None
        
        gdf_gt = load_gt(date, TARGET_CRS)
        gt_data_for_triple_plot["gdf"] = gdf_gt

        is_gt_valid = not gdf_gt.empty and gdf_gt.geometry.is_valid.all()
        if not is_gt_valid:
            logger.warning(f"Invalid or empty ground truth for date {date}. Treating as 0% risk day.")
            gdf_gt = gpd.GeoDataFrame(columns=['risk_level', 'geometry'], crs=TARGET_CRS)
            gt_data_for_triple_plot["gdf"] = gdf_gt

        current_day_max_gt_risk = get_max_risk_level(gdf_gt)
        current_day_has_gt_risk = (current_day_max_gt_risk != '0%')

        gdf_spc = load_spc(date, TARGET_CRS)
        llm_predictions = load_llm(date, TARGET_CRS)

        if model_filter:
            llm_predictions = {k: v for k, v in llm_predictions.items() if k in model_filter}

        models_to_evaluate = {}
        if gdf_spc is not None:
            if gdf_spc.geometry.is_valid.all():
                 models_to_evaluate['SPC'] = gdf_spc
            else:
                 logger.warning(f"Skipping SPC prediction due to invalid geometries for date {date}")

        for model_name_llm, gdf_llm in llm_predictions.items():
             if gdf_llm.geometry.is_valid.all():
                  models_to_evaluate[model_name_llm] = gdf_llm
             else:
                  logger.warning(f"Skipping LLM prediction for {model_name_llm} due to invalid geometries on date {date}")

        if not models_to_evaluate:
            logger.warning(f"No valid predictions found to evaluate for date {date}")
            continue

        for model_name, gdf_pred in models_to_evaluate.items():
            logger.info(f"  Evaluating model: {model_name}")

            try:
                ious, union_areas, geom_gt_nonzero, geom_pred_nonzero = compute_geometric_iou(gdf_gt, gdf_pred, domain_polygon)

                gt_overall_centroid_x, gt_overall_centroid_y = None, None
                if geom_gt_nonzero and not geom_gt_nonzero.is_empty:
                    centroid = geom_gt_nonzero.centroid
                    if not centroid.is_empty:
                        gt_overall_centroid_x, gt_overall_centroid_y = centroid.x, centroid.y

                pred_overall_centroid_x, pred_overall_centroid_y = None, None
                overall_centroid_distance = None
                if geom_pred_nonzero and not geom_pred_nonzero.is_empty:
                    centroid = geom_pred_nonzero.centroid
                    if not centroid.is_empty:
                        pred_overall_centroid_x, pred_overall_centroid_y = centroid.x, centroid.y

                        if gt_overall_centroid_x is not None and gt_overall_centroid_y is not None and \
                           pred_overall_centroid_x is not None and pred_overall_centroid_y is not None:
                            gt_point = Point(gt_overall_centroid_x, gt_overall_centroid_y)
                            pred_point = Point(pred_overall_centroid_x, pred_overall_centroid_y)
                            overall_centroid_distance = gt_point.distance(pred_point)
                
                gt_hr_centroid_x, gt_hr_centroid_y = None, None
                if current_day_max_gt_risk != '0%':
                    valid_gt_hr_geoms = gdf_gt[(gdf_gt['risk_level'] == current_day_max_gt_risk) & gdf_gt.geometry.is_valid].geometry
                    if not valid_gt_hr_geoms.empty:
                        geom_gt_hr = unary_union(valid_gt_hr_geoms.values)
                        if geom_gt_hr and not geom_gt_hr.is_empty:
                            centroid_hr_gt = geom_gt_hr.centroid
                            if not centroid_hr_gt.is_empty:
                                gt_hr_centroid_x, gt_hr_centroid_y = centroid_hr_gt.x, centroid_hr_gt.y
                    else:
                        logger.debug(f"No valid geometries for GT highest risk level {current_day_max_gt_risk} on {date}")

                pred_hr_centroid_x, pred_hr_centroid_y = None, None
                hr_centroid_distance = None
                max_risk_pred_level = get_max_risk_level(gdf_pred)
                if max_risk_pred_level != '0%':
                    valid_pred_hr_geoms = gdf_pred[(gdf_pred['risk_level'] == max_risk_pred_level) & gdf_pred.geometry.is_valid].geometry
                    if not valid_pred_hr_geoms.empty:
                        geom_pred_hr = unary_union(valid_pred_hr_geoms.values)
                        if geom_pred_hr and not geom_pred_hr.is_empty:
                            centroid_hr_pred = geom_pred_hr.centroid
                            if not centroid_hr_pred.is_empty:
                                pred_hr_centroid_x, pred_hr_centroid_y = centroid_hr_pred.x, centroid_hr_pred.y
                                if gt_hr_centroid_x is not None and gt_hr_centroid_y is not None and \
                                   pred_hr_centroid_x is not None and pred_hr_centroid_y is not None:
                                    gt_hr_point = Point(gt_hr_centroid_x, gt_hr_centroid_y)
                                    pred_hr_point = Point(pred_hr_centroid_x, pred_hr_centroid_y)
                                    hr_centroid_distance = gt_hr_point.distance(pred_hr_point)
                    else:
                        logger.debug(f"No valid geometries for Pred highest risk level {max_risk_pred_level} on {date} for model {model_name}")

                max_risk_pred = get_max_risk_level(gdf_pred)
                mean_iou, tb_score, weight, is_false_alarm, is_hard_hallucination, daily_weighted_hallucination_loss = daily_scores(ious, union_areas, gdf_gt, gdf_pred, geom_gt_nonzero, geom_pred_nonzero, max_risk_pred)

                try:
                    gt_idx = RISK_CATEGORIES.index(current_day_max_gt_risk)
                    pred_idx = RISK_CATEGORIES.index(max_risk_pred)
                    if pred_idx < gt_idx:
                        max_risk_status = 'under'
                    elif pred_idx == gt_idx:
                        max_risk_status = 'match'
                    else:
                        max_risk_status = 'over'
                except ValueError as e:
                    logger.error(f"Error comparing max risk levels ('{current_day_max_gt_risk}', '{max_risk_pred}'): {e}")
                    max_risk_status = 'error'

                result = {
                    "date": date,
                    "model": model_name,
                    "per_category_iou": ious,
                    "daily_mean_iou": mean_iou,
                    "daily_tornadobench": tb_score,
                    "daily_weight": weight,
                    "is_false_alarm": is_false_alarm,
                    "is_hard_hallucination": is_hard_hallucination,
                    "max_risk_status": max_risk_status,
                    "daily_weighted_hallucination_loss": daily_weighted_hallucination_loss,
                    "gt_overall_centroid_x": gt_overall_centroid_x,
                    "gt_overall_centroid_y": gt_overall_centroid_y,
                    "pred_overall_centroid_x": pred_overall_centroid_x,
                    "pred_overall_centroid_y": pred_overall_centroid_y,
                    "overall_centroid_distance": overall_centroid_distance,
                    "gt_hr_centroid_x": gt_hr_centroid_x,
                    "gt_hr_centroid_y": gt_hr_centroid_y,
                    "pred_hr_centroid_x": pred_hr_centroid_x,
                    "pred_hr_centroid_y": pred_hr_centroid_y,
                    "hr_centroid_distance": hr_centroid_distance,
                    "gt_had_actual_risk": current_day_has_gt_risk
                }
                all_results.append(result)

                json_path = f"iou_results/jsons/iou_results_{model_name}_{date}.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

                if date in paper_submission_cases and gt_data_for_triple_plot["overall_coords"] is None:
                    gt_data_for_triple_plot["overall_coords"] = (gt_overall_centroid_x, gt_overall_centroid_y)
                    gt_data_for_triple_plot["hr_coords"] = (gt_hr_centroid_x, gt_hr_centroid_y)

                if not args.skip_plots:
                    plot_vector_comparison(gdf_gt, gdf_pred, date, model_name, TARGET_CRS, tb_score,
                                           gt_overall_centroid_coords=(gt_overall_centroid_x, gt_overall_centroid_y),
                                           pred_overall_centroid_coords=(pred_overall_centroid_x, pred_overall_centroid_y),
                                           gt_hr_centroid_coords=(gt_hr_centroid_x, gt_hr_centroid_y),
                                           pred_hr_centroid_coords=(pred_hr_centroid_x, pred_hr_centroid_y))

                if date in paper_submission_cases:
                    if model_name == 'SPC':
                        spc_data_for_triple_plot = {
                            "gdf": gdf_pred.copy(), "name": "SPC", "tb_score": tb_score,
                            "overall_coords": (pred_overall_centroid_x, pred_overall_centroid_y) if pred_overall_centroid_x is not None else (None,None),
                            "hr_coords": (pred_hr_centroid_x, pred_hr_centroid_y) if pred_hr_centroid_x is not None else (None,None)
                        }
                    elif model_name == paper_submission_cases[date]:
                        target_model_data_for_triple_plot = {
                            "gdf": gdf_pred.copy(), "name": model_name, "tb_score": tb_score,
                            "overall_coords": (pred_overall_centroid_x, pred_overall_centroid_y) if pred_overall_centroid_x is not None else (None,None),
                            "hr_coords": (pred_hr_centroid_x, pred_hr_centroid_y) if pred_hr_centroid_x is not None else (None,None)
                        }

                if not args.skip_plots and date not in paper_submission_cases:
                    plot_vector_comparison(gdf_gt, gdf_pred, date, model_name, TARGET_CRS, tb_score,
                                           gt_overall_centroid_coords=(gt_overall_centroid_x, gt_overall_centroid_y),
                                           pred_overall_centroid_coords=(pred_overall_centroid_x, pred_overall_centroid_y),
                                           gt_hr_centroid_coords=(gt_hr_centroid_x, gt_hr_centroid_y),
                                           pred_hr_centroid_coords=(pred_hr_centroid_x, pred_hr_centroid_y))

            except Exception as e:
                logger.error(f"Error processing model {model_name} for date {date}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        if date in paper_submission_cases and not args.skip_plots:
            if gt_data_for_triple_plot["gdf"] is not None and spc_data_for_triple_plot is not None and target_model_data_for_triple_plot is not None:
                logger.info(f"Generating triple plot for date {date} with SPC and {target_model_data_for_triple_plot['name']}")
                plot_vector_comparison_triple(
                    gdf_gt=gt_data_for_triple_plot["gdf"],
                    gdf_spc=spc_data_for_triple_plot["gdf"],
                    gdf_model=target_model_data_for_triple_plot["gdf"],
                    date=date,
                    model_name_spc="SPC",
                    model_name_model=target_model_data_for_triple_plot["name"],
                    target_crs=TARGET_CRS,
                    tornadobench_score_spc=spc_data_for_triple_plot["tb_score"],
                    tornadobench_score_model=target_model_data_for_triple_plot["tb_score"],
                    gt_overall_centroid_coords=gt_data_for_triple_plot["overall_coords"],
                    spc_overall_centroid_coords=spc_data_for_triple_plot["overall_coords"],
                    model_overall_centroid_coords=target_model_data_for_triple_plot["overall_coords"],
                    gt_hr_centroid_coords=gt_data_for_triple_plot["hr_coords"],
                    spc_hr_centroid_coords=spc_data_for_triple_plot["hr_coords"],
                    model_hr_centroid_coords=target_model_data_for_triple_plot["hr_coords"]
                )
            else:
                logger.warning(f"Could not generate triple plot for {date} due to missing data components.")
                if not args.skip_plots:
                    if spc_data_for_triple_plot:
                         plot_vector_comparison(gt_data_for_triple_plot["gdf"], spc_data_for_triple_plot["gdf"], date, "SPC", TARGET_CRS, spc_data_for_triple_plot["tb_score"],
                                               gt_overall_centroid_coords=gt_data_for_triple_plot["overall_coords"],
                                               pred_overall_centroid_coords=spc_data_for_triple_plot["overall_coords"],
                                               gt_hr_centroid_coords=gt_data_for_triple_plot["hr_coords"],
                                               pred_hr_centroid_coords=spc_data_for_triple_plot["hr_coords"])
                    if target_model_data_for_triple_plot:
                        plot_vector_comparison(gt_data_for_triple_plot["gdf"], target_model_data_for_triple_plot["gdf"], date, target_model_data_for_triple_plot["name"], TARGET_CRS, target_model_data_for_triple_plot["tb_score"],
                                              gt_overall_centroid_coords=gt_data_for_triple_plot["overall_coords"],
                                              pred_overall_centroid_coords=target_model_data_for_triple_plot["overall_coords"],
                                              gt_hr_centroid_coords=gt_data_for_triple_plot["hr_coords"],
                                              pred_hr_centroid_coords=target_model_data_for_triple_plot["hr_coords"])

    if not all_results:
        logger.error("No results generated")
        return

    df_results = pd.DataFrame(all_results)

    daily_scores_path = "iou_results/daily_model_scores.csv"
    try:
        logger.info(f"Saving daily scores to {daily_scores_path}")
        df_results.to_csv(daily_scores_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save daily scores: {e}")

    unique_day_weights = df_results[['date', 'daily_weight']].drop_duplicates()
    common_denominator = unique_day_weights['daily_weight'].sum()

    if common_denominator <= 0:
        logger.warning("Common denominator (total weight across all days) is zero. Cannot calculate normalized TornadoBench scores.")

    summary_rows = []
    for model, group in df_results.groupby('model'):
        weighted_sum = (group['daily_tornadobench'] * group['daily_weight']).sum()
        overall_tb = weighted_sum / common_denominator if common_denominator > 0 else 0.0

        false_alarm_days = group['is_false_alarm'].sum()
        total_days = len(group)
        tornado_hallucination_simple_score = false_alarm_days / total_days if total_days > 0 else 0.0

        overall_weighted_hallucination_loss = group['daily_weighted_hallucination_loss'].mean()

        status_counts = group['max_risk_status'].value_counts()
        under_count = status_counts.get('under', 0)
        match_count = status_counts.get('match', 0)
        over_count = status_counts.get('over', 0)
        error_count = status_counts.get('error', 0)
        if error_count > 0:
            logger.warning(f"Model '{model}' had {error_count} days with max risk status errors.")
        
        total_valid_days = total_days - error_count
        if total_valid_days > 0:
            under_rate = (under_count / total_valid_days) * 100
            match_rate = (match_count / total_valid_days) * 100
            over_rate = (over_count / total_valid_days) * 100
        else:
            under_rate, match_rate, over_rate = 0.0, 0.0, 0.0

        avg_gt_overall_centroid_x = group['gt_overall_centroid_x'].dropna().mean()
        avg_gt_overall_centroid_y = group['gt_overall_centroid_y'].dropna().mean()
        avg_pred_overall_centroid_x = group['pred_overall_centroid_x'].dropna().mean()
        avg_pred_overall_centroid_y = group['pred_overall_centroid_y'].dropna().mean()
        avg_overall_centroid_distance = group['overall_centroid_distance'].dropna().mean()

        avg_gt_hr_centroid_x = group['gt_hr_centroid_x'].dropna().mean()
        avg_gt_hr_centroid_y = group['gt_hr_centroid_y'].dropna().mean()
        avg_pred_hr_centroid_x = group['pred_hr_centroid_x'].dropna().mean()
        avg_pred_hr_centroid_y = group['pred_hr_centroid_y'].dropna().mean()
        avg_hr_centroid_distance = group['hr_centroid_distance'].dropna().mean()

        score_formula_parts = []
        total_score_check = 0.0

        weight_to_level = {v: k for k, v in RISK_WEIGHTS.items()}

        grouped_by_weight = group.groupby('daily_weight')

        for weight_val, weight_group in grouped_by_weight:
            level = weight_to_level.get(weight_val, f"UnknownWeight({weight_val})")
            num_days = len(weight_group)
            sum_scores = weight_group['daily_tornadobench'].sum()
            
            score_formula_parts.append(f"({sum_scores:.4f} * {weight_val})")
            
            total_score_check += sum_scores * weight_val

        total_score_formula = " + ".join(score_formula_parts)
        
        if not np.isclose(total_score_check, weighted_sum):
             logger.warning(f"Score calculation mismatch for {model}: {total_score_check} vs {weighted_sum}")

        summary_row_data = {
            "model": model,
            "TornadoBench": overall_tb,
            "TornadoHallucinationSimple": tornado_hallucination_simple_score,
            "TornadoHallucinationHard": overall_weighted_hallucination_loss,
            "UnderforecastRate": under_rate,
            "MatchRate": match_rate,
            "OverforecastRate": over_rate,
            "CalculatedWeightedSum": weighted_sum,
            "TotalScoreFormula": total_score_formula,
            "CommonDenominator": common_denominator,
            "AvgGTOverallCentroidX": avg_gt_overall_centroid_x if not np.isnan(avg_gt_overall_centroid_x) else None,
            "AvgGTOverallCentroidY": avg_gt_overall_centroid_y if not np.isnan(avg_gt_overall_centroid_y) else None,
            "AvgPredOverallCentroidX": avg_pred_overall_centroid_x if not np.isnan(avg_pred_overall_centroid_x) else None,
            "AvgPredOverallCentroidY": avg_pred_overall_centroid_y if not np.isnan(avg_pred_overall_centroid_y) else None,
            "AvgOverallCentroidDistance": avg_overall_centroid_distance if not np.isnan(avg_overall_centroid_distance) else None,
            "AvgGTHRCentroidX": avg_gt_hr_centroid_x if not np.isnan(avg_gt_hr_centroid_x) else None,
            "AvgGTHRCentroidY": avg_gt_hr_centroid_y if not np.isnan(avg_gt_hr_centroid_y) else None,
            "AvgPredHRCentroidX": avg_pred_hr_centroid_x if not np.isnan(avg_pred_hr_centroid_x) else None,
            "AvgPredHRCentroidY": avg_pred_hr_centroid_y if not np.isnan(avg_pred_hr_centroid_y) else None,
            "AvgHRCentroidDistance": avg_hr_centroid_distance if not np.isnan(avg_hr_centroid_distance) else None
        }
        summary_rows.append(summary_row_data)

    summary_columns = [
        "model", "TornadoBench", 
        "TornadoHallucinationSimple",
        "TornadoHallucinationHard",
        "UnderforecastRate", "MatchRate", "OverforecastRate",
        "CalculatedWeightedSum",
        "TotalScoreFormula",
        "CommonDenominator",
        "AvgGTOverallCentroidX", "AvgGTOverallCentroidY", 
        "AvgPredOverallCentroidX", "AvgPredOverallCentroidY", "AvgOverallCentroidDistance",
        "AvgGTHRCentroidX", "AvgGTHRCentroidY",
        "AvgPredHRCentroidX", "AvgPredHRCentroidY", "AvgHRCentroidDistance" # New centroid columns
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_columns)

    # Sort by TornadoBench score
    df_summary.sort_values("TornadoBench", ascending=False, inplace=True)

    # Save summary to CSV
    # Using %.4f to save with four decimal places for scores in 0-1 scale
    df_summary.to_csv("iou_results/model_summary.csv", index=False, float_format="%.4f")

    # Generate summary plots
    if not args.skip_plots:
        # Pass the relevant columns for plotting
        plot_df = df_summary[['model', 
                              'TornadoBench', 
                              'TornadoHallucinationSimple', 
                              'TornadoHallucinationHard']].copy()
        # Multiply by 100 for percentage display in plots
        plot_df['TornadoBench'] = plot_df['TornadoBench'] * 100
        # TornadoHallucinationSimple is already in percentage
        plot_summary_bars(plot_df)

        # Generate Max Risk Match plot
        plot_max_risk_match_summary(df_summary) # Pass the full summary df

        # Generate temporal consistency plot
        plot_temporal_consistency(df_results) # Pass the daily results for temporal plot

    # Print summary in the desired formula format
    print("\n--- Final Model Summary (TornadoBench Calculation) ---")
    
    # Loop through df_summary AGAIN to print the results
    # This ensures we use the finalized scores calculated with the common denominator
    for index, row in df_summary.iterrows():
        # Print using the CORRECTED score (overall_tb_original, now calculated with common denominator)
        # and the COMMON denominator info.
        print(f"Model: {row['model']}")
        print(f"  TornadoBench Score: {row['TornadoBench']*100:.2f}%") # Use the recalculated score from df_summary
        print(f"  TornadoHallucinationSimple: {row['TornadoHallucinationSimple']:.2f} (Lower is Better)") # Modified to .2f from .4f
        print(f"  TornadoHallucinationHard: {row['TornadoHallucinationHard']:.2f} (Lower is Better)") # Modified to .2f from .4f
        print(f"  Max Risk: {row['UnderforecastRate']:.2f}% Under / {row['MatchRate']:.2f}% Match / {row['OverforecastRate']:.2f}% Over") # Modified to .2f from .1f
        
        # Print Centroid Information (convert distance from meters to km for readability if large)
        gt_overall_cx_str = f"{row['AvgGTOverallCentroidX']:.0f}" if pd.notna(row['AvgGTOverallCentroidX']) else "N/A"
        gt_overall_cy_str = f"{row['AvgGTOverallCentroidY']:.0f}" if pd.notna(row['AvgGTOverallCentroidY']) else "N/A"
        pred_overall_cx_str = f"{row['AvgPredOverallCentroidX']:.0f}" if pd.notna(row['AvgPredOverallCentroidX']) else "N/A"
        pred_overall_cy_str = f"{row['AvgPredOverallCentroidY']:.0f}" if pd.notna(row['AvgPredOverallCentroidY']) else "N/A"
        avg_overall_dist_str = f"{int(round(row['AvgOverallCentroidDistance']/1000))} km" if pd.notna(row['AvgOverallCentroidDistance']) else "N/A"

        gt_hr_cx_str = f"{row['AvgGTHRCentroidX']:.0f}" if pd.notna(row['AvgGTHRCentroidX']) else "N/A"
        gt_hr_cy_str = f"{row['AvgGTHRCentroidY']:.0f}" if pd.notna(row['AvgGTHRCentroidY']) else "N/A"
        pred_hr_cx_str = f"{row['AvgPredHRCentroidX']:.0f}" if pd.notna(row['AvgPredHRCentroidX']) else "N/A"
        pred_hr_cy_str = f"{row['AvgPredHRCentroidY']:.0f}" if pd.notna(row['AvgPredHRCentroidY']) else "N/A"
        avg_hr_dist_str = f"{int(round(row['AvgHRCentroidDistance']/1000))} km" if pd.notna(row['AvgHRCentroidDistance']) else "N/A"
        
        # Remove coordinate display, just keep distances
        print(f"  Avg Overall Risk Centroid Distance: {avg_overall_dist_str}")
        print(f"  Avg Max Risk Centroid Distance: {avg_hr_dist_str}")
        
        # Use the specific model's score formula but the common denominator formula/value
        print(f"  Numerator = Sum(DailyTornadoBench * Weight) = {row['TotalScoreFormula']} = {row['CalculatedWeightedSum']:.2f}") # Modified to .2f from .4f
        print(f"  Score (0-1) = Numerator / Denominator = {row['CalculatedWeightedSum']:.2f} / {common_denominator:.2f} = {row['TornadoBench']:.2f}") # Modified all to .2f
        print(f"  Score (%) = Score (0-1) * 100 = {row['TornadoBench']*100:.2f}%")
        print()

    # --- Common Denominator Printout ---
    # Recalculate the common denominator formula based on ALL days, not just the first model's days
    common_denominator_formula_parts = []
    common_denominator_weight_check = 0.0
    weight_to_level_map = {v: k for k, v in RISK_WEIGHTS.items()} # Ensure this map is available or recalculate
    
    # Get unique day-weight pairs first to avoid double counting if multiple models ran on same day
    # unique_day_weights = df_results[['date', 'daily_weight']].drop_duplicates() # Moved this calculation up
    
    # Group these unique day-weights by weight value
    grouped_weights_all_days = unique_day_weights.groupby('daily_weight')

    for weight_val, weight_group in grouped_weights_all_days:
        level = weight_to_level_map.get(weight_val, f"UnknownWeight({weight_val})")
        num_days = len(weight_group)
        common_denominator_formula_parts.append(f"({num_days} * {weight_val})")
        common_denominator_weight_check += num_days * weight_val
        
    common_denominator_formula = " + ".join(common_denominator_formula_parts)
    
    if not np.isclose(common_denominator_weight_check, common_denominator):
        logger.warning(f"Common Denominator calculation mismatch: Formula sum {common_denominator_weight_check} vs Direct sum {common_denominator}")

    print("--- Common Denominator ---")
    print(f"  Denominator = Sum(Weight) = {common_denominator_formula} = {common_denominator:.2f}")
    print("\nSummary also saved to iou_results/model_summary.csv with formula components.")
    print()


def _smoke_test():
    print("Running smoke test...")
    try:
        domain = create_domain_polygon()
        assert domain is not None and domain.is_valid
        print("Domain creation OK.")

        date_dirs = sorted(glob.glob("ppf_output/????????"))
        if not date_dirs:
            print("No PPF output found; smoke test skipped further checks.")
            return
        test_date = os.path.basename(date_dirs[0])

        gdf_gt_test = load_gt(test_date, TARGET_CRS)
        assert isinstance(gdf_gt_test, gpd.GeoDataFrame)
        print(f"GT loading for {test_date} OK (returned GDF).")

        print("Smoke test passed basic checks.")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()