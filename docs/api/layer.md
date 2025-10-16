# YirgacheffeLayer

A layer object represents some geospatial data. This might come from:

* A raster data source, for example, data from a GeoTIFF
* A polygon data source, for example, data from a GPKG, GeoJSON, or ESRI Shapefile
* A constant value
* An expression built up using the above inputs using operators to add, divide, etc.

::: yirgacheffe.YirgacheffeLayer
    handler: python
    options:
        members:
            - area
            - datatype
            - latlng_for_pixel
            - map_projection
            - pixel_for_latlng
            - read_array
            - show
            - to_geotiff
        show_root_heading: true
        heading_level: 2
        show_source: false
        show_bases: false
        inherited_members: true
        filters:
            - "!__"
