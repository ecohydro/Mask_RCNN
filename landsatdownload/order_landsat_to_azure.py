import download_utils as du
import datetime
import yaml
import time
import click


@click.command()
@click.argument("config_path")
def run(config_path):
    """
    Runs the download process for azure. Reads the config file in order
    to subset the order by geographic bounds, date, or path row.

    Args:
        config_path (str): Takes the path to the config file, which contains 
        credentials for azure, storage paths, and other info (see template).

    Returns: Nothing is returned. Used for its side effect of downloading to azure.

    """

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    bbox = du.get_bbox_from_wbd(
        configs["wbd_gdb_path"],
        configs["huc_level"],
        str(configs["huc_id"])
    )

    scene_list = du.get_scene_list(
        collection=configs["collection"],
        bbox=bbox,
        begin=datetime.datetime(
            configs["year_start"],
            configs["month_start"],
            configs["day_start"],
        ),
        end=datetime.datetime(
            configs["year_end"],
            configs["month_end"],
            configs["day_end"],
        ),
        max_results=configs["max_results"],
        max_cloud_cover=configs["max_cloud_cover"],
    )

    pathrow_list = configs["path_row_list"]

    scene_list = du.filter_scenes_by_path_row(scene_list, pathrow_list)

    product_list = configs["product_list"]

    order = du.submit_order(scene_list, product_list)
    print("Order status: " + order.status)
    print("Order ID: " + order.orderid)
    du.azure_download_order(order, configs)
    print("Order status: " + order.status)
    print("Order ID: " + order.orderid)


run()
