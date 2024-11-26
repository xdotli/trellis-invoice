import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import ast
from Levenshtein import distance as levenshtein_distance
from jiwer import wer

predefined_headers = [
        "invoice_date",
        "seller",
        "invoice_no",
        "client",
        "iban",
        "seller_tax_id",
        "summary",
        "client_tax_id",
        "items"
    ]

def api_request(method, url, api_key, payload=None):
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    response = requests.request(method, url, json=payload, headers=headers)
    return response

def create_project(proj_name, api_key):
    url = "https://api.runtrellis.com/v1/projects/create"
    payload = {"name": proj_name}
    response = api_request("POST", url, api_key, payload)
    print("create_project", response)
    return response.json()["data"]["proj_id"]

def load_transformation(api_key, proj_id, json_file="invoices-donut-data-v1.json"):
    with open(json_file, "r") as f:
        transform_params = json.load(f)
    url = "https://api.runtrellis.com/v1/transforms/create"
    payload = {
        "proj_id": proj_id,
        "transform_name": "email_analysis",
        "transform_params": transform_params
    }
    response = api_request("POST", url, api_key, payload)
    print("load_transformation", response)
    return response.json()["data"]["transform_id"]

def create_event_trigger(proj_id, transform_id, api_key):
    url = "https://api.runtrellis.com/v1/events/subscriptions/actions/bulk"
    payload = {
        "events_with_actions": [
            {
                "event_type": "asset_uploaded",
                "proj_id": proj_id,
                "actions": [{"type": "run_extraction", "proj_id": proj_id}]
            },
            {
                "event_type": "asset_extracted",
                "proj_id": proj_id,
                "actions": [{"type": "refresh_transform", "transform_id": transform_id}]
            }
        ]
    }
    response = api_request("POST", url, api_key, payload)
    print("create_enevnt", response)

def upload_data(proj_id, api_key, urls):
    url = "https://api.runtrellis.com/v1/assets/upload"
    payload = {
        "proj_id": proj_id,
        "urls": urls
    }
    response = api_request("POST", url, api_key, payload)
    print("upload_data", response)
    return [data["asset_id"] for data in response.json()["data"]]

def get_result(asset_ids, api_key, transform_id):
    url = f"https://api.runtrellis.com/v1/transforms/{transform_id}/results"
    payload = {"filters": {}, "asset_ids": asset_ids}
    max_retries = 5

    for i in range(max_retries):
        try:
            response = api_request("POST", url, api_key, payload)
            print("get_result", response)

            if 500 <= response.status_code < 600:
                print(f"Received {response.status_code} response, retry in 5 seconds...")
                time.sleep(5)
                continue

            response.raise_for_status()
            response_data = response.json()
            temp_df = pd.DataFrame(columns=predefined_headers)
            res_df = post_process("invoice", response_data, temp_df)
            return res_df
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}, retry after 5 seconds...")
            time.sleep(5)

    print("Maximum retries reached. Returning empty DataFrame.")
    return pd.DataFrame(columns=predefined_headers, index=range(len(asset_ids)))

def post_process_invoice_data(input_json, df):
    operations = {
        col["id"]: col["name"]
        for col in input_json["metadata"]["column_definitions"]
        if col["group"] == "operation"
    }

    result_dicts = []
    for operation_data in input_json["data"]:
        result_dict = {
            operations[op_id]: operation_data[op_id]
            for op_id in operations
            if op_id in operation_data
        }
        summary_dict = json.loads(result_dict['summary'])
        items_list = json.loads(result_dict['items'])

        result_dict['summary'] = summary_dict
        result_dict['items'] = items_list

        result_dicts.append(result_dict)

    new_rows = pd.DataFrame(result_dicts)
    return pd.concat([df, new_rows], ignore_index=True)

def post_process(method, response_data, batch_df):
    if method == "invoice":
        return post_process_invoice_data(response_data, batch_df)
    else:
        pass

def generate_urls(base_url, start_index, num_images):
    file_prefix = "image_"
    file_extension = ".JPEG"
    urls = []
    for i in range(start_index, start_index + num_images):
        file_name = f"{file_prefix}{i}{file_extension}"
        urls.append(f"{base_url}{file_name}")
    return urls

def extract_fields(json_str):
    data = ast.literal_eval(json_str)
    gt_parse = data.get('gt_parse', {})
    header = gt_parse.get('header', {})
    summary = gt_parse.get('summary', {})
    items = gt_parse.get('items', [])

    result = {
        'invoice_date': header.get('invoice_date'),
        'seller': header.get('seller'),
        'invoice_no': header.get('invoice_no'),
        'client': header.get('client'),
        'iban': header.get('iban'),
        'seller_tax_id': header.get('seller_tax_id'),
        'summary': summary,
        'client_tax_id': header.get('client_tax_id'),
        'items': items
    }
    return pd.Series(result)

def eval(transformed_df, groundtruth_df):
    if not list(transformed_df.columns) == list(groundtruth_df.columns):
        raise ValueError("The columns of transformed and groundtruth DataFrames must match.")

    column_metrics = []

    for column in transformed_df.columns:
        transformed_col = transformed_df[column]
        groundtruth_col = groundtruth_df[column]

        metrics = {"Column": column, "Average WER": 0, "Average Levenshtein": 0}
        wer_scores = []
        lev_distances = []

        for text1, text2 in zip(transformed_col, groundtruth_col):
            wer_scores.append(wer(str(text1), str(text2)))
            lev_distances.append(levenshtein_distance(str(text1), str(text2)))

        metrics["Average WER"] = sum(wer_scores) / len(wer_scores)
        metrics["Average Levenshtein"] = sum(lev_distances) / len(lev_distances)

        column_metrics.append(metrics)


    return pd.DataFrame(column_metrics)

def visual(metrics_df):
    num_metrics = metrics_df.shape[1] - 1
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 6 * num_metrics), constrained_layout=True)

    for i, metric in enumerate(metrics_df.columns[1:]):
        heatmap_data = metrics_df.set_index("Column")[[metric]]

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": metric},
            ax=axes[i]
        )
        axes[i].set_title(f"Heatmap of {metric}")
        axes[i].set_ylabel("Columns")

    plt.savefig("heatmap.jpg", bbox_inches="tight", dpi=300)

def setup_project(api_key, project_name="test2"):
    proj_id = create_project(project_name, api_key)
    transform_id = load_transformation(api_key, proj_id)
    create_event_trigger(proj_id, transform_id, api_key)
    return proj_id, transform_id

def process_batches(proj_id, api_key, transform_id, base_url, total_images, batch_size):
    extract_df = pd.DataFrame(columns=predefined_headers)
    for start_index in range(0, total_images, batch_size):
        num_images = min(batch_size, total_images - start_index)
        urls = generate_urls(base_url, start_index, num_images)
        print(urls)
        asset_ids = upload_data(proj_id, api_key, urls)
        print(asset_ids)
        batch_df = get_result(asset_ids, api_key, transform_id)
        batch_df = batch_df.replace({r'\n': ' '}, regex=True)
        extract_df = pd.concat([extract_df, batch_df], ignore_index=True)
    return extract_df

def save_to_csv(data, filename):
    data.to_csv(filename, index=False)

def load_ground_truth(split_key="test"):
    splits = {
        'train': 'data/train-00000-of-00001-a5c51039eab2980a.parquet',
        'validation': 'data/validation-00000-of-00001-b8a5c4a6237baf25.parquet',
        'test': 'data/test-00000-of-00001-56af6bd5ff7eb34d.parquet'
    }
    ground_truth = pd.read_parquet("hf://datasets/katanaml-org/invoices-donut-data-v1/" + splits[split_key])
    ground_truth = ground_truth[["ground_truth"]]
    ground_truth = ground_truth['ground_truth'].apply(extract_fields)
    return ground_truth

def main():
    YOUR_API_KEY = os.getenv("TRELLIS_TOKEN")
    proj_id, transform_id = setup_project(YOUR_API_KEY)

    base_url = "https://hub.oxen.ai/api/repos/katanaml-org/invoices-donut-data-v1/file/main/images/train/"
    total_images = 425
    batch_size = 10
    extract_df = process_batches(proj_id, YOUR_API_KEY, transform_id, base_url, total_images, batch_size)
    extract_df.to_csv("extract.csv", index=False)
    ground_truth = load_ground_truth("test")

    data_vis = eval(extract_df, ground_truth)
    print(data_vis)
    visual(data_vis)

if __name__ == "__main__":
    main()
