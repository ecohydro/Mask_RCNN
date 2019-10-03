from pathlib import Path
import pandas as pd 
import xml.etree.ElementTree as et 

def product_table_from_unpacked_archives(ard_folder="/mnt/cropmaskperm/unpacked_ard_landsat_downloads/ARDSR/", xml_folder="/mnt/cropmaskperm/unpacked_ard_landsat_downloads/ARDxml/"):

    root_dir_sr = Path(ard_folder)
    root_dir_xml = Path(xml_folder)

    scene_paths = sorted(root_dir_sr.glob("*"))
    xml_paths = sorted(root_dir_xml.glob("*"))
    df_cols = ["cloud_cover", "cloud_shadow", "snow_ice", "fill", "instrument", "level1_collection", "ard_version"]
    rows = []

    for xml_path in xml_paths:

        xtree = et.parse(xml_path)
        tile_meta_global = list(xtree.getroot())[0][0]
        dataframe_dict = {}

        element = tile_meta_global.find("{https://landsat.usgs.gov/ard/v1}"+"tile_grid")
        h = element.attrib['h']
        v = element.attrib['v']

        element = tile_meta_global.find("{https://landsat.usgs.gov/ard/v1}"+"acquisition_date")
        datetime = pd.to_datetime(element.text, format="%Y-%m-%d")

        dataframe_dict.update({'h':h, 'v':v, 'acquisition_date':datetime})

        for col in df_cols:
            element = tile_meta_global.find("{https://landsat.usgs.gov/ard/v1}"+col)
            if col in ["cloud_cover", "cloud_shadow", "snow_ice", "fill"]:
                element.text = float(element.text)
            dataframe_dict.update({col:element.text})
        rows.append(dataframe_dict)

    out_df = pd.DataFrame(rows, columns = df_cols.extend(['acquisition_date', 'h','v']))

    out_df = out_df.set_index("acquisition_date")

    out_df['xml_paths'] = xml_paths
    out_df['scene_paths'] = scene_paths
    
    return out_df