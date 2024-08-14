# glidergun

```
pip install glidergun
```

![SRTM](srtm.png)


## creation / io

- grid
- stack
- save

## operators

- most overloadable Python operators

## properties

- width
- height
- dtype
- xmin
- ymin
- xmax
- ymax
- extent
- mean
- std
- min
- max
- cell_size
- bins
- md5

## local

- local (higher order)
- is_nan
- abs
- sin
- cos
- tan
- arcsin
- arccos
- arctan
- log
- round
- gaussian_filter
- gaussian_gradient_magnitude
- gaussian_laplace
- prewitt
- sobel
- uniform_filter
- uniform_filter1d

## focal

- focal (higher order)
- focal_count
- focal_mean
- focal_std
- focal_var
- focal_median
- focal_min
- focal_max
- focal_sum
- focal_ptp
- focal_percentile
- focal_quantile
- focal_entropy
- focal_hmean
- focal_pmean
- focal_kurtosis
- focal_iqr
- focal_mode
- focal_moment
- focal_skew
- focal_kstat
- focal_kstatvar
- focal_tmean
- focal_tvar
- focal_tmin
- focal_tmax
- focal_tstd
- focal_variation
- focal_median_abs_deviation
- focal_chisquare
- focal_ttest_ind

## zonal

- zonal (higher order)
- zonal_count
- zonal_mean
- zonal_std
- zonal_var
- zonal_median
- zonal_min
- zonal_max
- zonal_sum
- zonal_ptp
- zonal_percentile
- zonal_quantile
- zonal_entropy
- zonal_hmean
- zonal_pmean
- zonal_kurtosis
- zonal_iqr
- zonal_mode
- zonal_moment
- zonal_skew
- zonal_kstat
- zonal_kstatvar
- zonal_tmean
- zonal_tvar
- zonal_tmin
- zonal_tmax
- zonal_tstd
- zonal_variation
- zonal_median_abs_deviation

## interpolation

- interpolate (higher order)
- interp_linear
- interp_nearest
- interp_rbf

## regression / classification

- fit (higher order)

## surface

- aspect
- slope
- hillshade

## conversion, etc.

- buffer
- clip
- con
- fill_nan
- from_polygons
- mosaic
- pca
- percent_clip
- project
- randomize
- reclass
- replace
- resample
- scale
- set_nan
- standardize
- to_points
- to_polygons
- to_stack

## ipython

- plot (Matplotlib)
- map (Folium)
