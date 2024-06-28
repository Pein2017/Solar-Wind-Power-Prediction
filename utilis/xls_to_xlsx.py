import os

import pandas as pd


def convert_xls_to_xlsx(root_folder):
    """
    Recursively traverse all folders under the given root_folder and convert .xls files to .xlsx files.

    Parameters:
    root_folder (str): The root folder to start the recursive search.
    """
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xls"):
                xls_file_path = os.path.join(subdir, file)
                xlsx_file_path = os.path.join(
                    subdir, file + "x"
                )  # Adding 'x' to create .xlsx extension

                try:
                    # Load the .xls file
                    data = pd.read_excel(xls_file_path)

                    # Save it as .xlsx file
                    data.to_excel(xlsx_file_path, index=False)

                    print(f"Converted: {xls_file_path} to {xlsx_file_path}")
                except Exception as e:
                    print(f"Failed to convert {xls_file_path}: {e}")


def delete_xls_files(root_folder):
    """
    Recursively traverse all folders under the given root_folder and delete .xls files.

    Parameters:
    root_folder (str): The root folder to start the recursive search.
    """
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".xls"):
                xls_file_path = os.path.join(subdir, file)

                try:
                    # Delete the .xls file
                    os.remove(xls_file_path)
                    print(f"Deleted: {xls_file_path}")
                except Exception as e:
                    print(f"Failed to delete {xls_file_path}: {e}")


if __name__ == "__main__":
    # Specify the root folder
    root_folder = "/data/Pein/Pytorch/Wind-Solar-Prediction/光伏场站数据"

    # Call the function
    convert_xls_to_xlsx(root_folder)

    # Call the function
    delete_xls_files(root_folder)
